"""Обобщение истории по смыслу запросов и география внутри категории.

История должна быть уже очищена от оценочных взаимодействий. Все дополнительные
признаки вычисляются без gold. E5 используется локально, без дообучения/API.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import json
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from retrieval import ROOT, FEATURES, History, normalize

RANK_FIELDS=['bm25','title_cos','char_cos','dense_cos','params_cos','bm25_geo',
             'dense_geo','char_geo','cat_probability','history_clicks']
CONTEXT_FIELDS=['semantic_cat_probability','semantic_history_clicks','semantic_history_maxsim',
                'semantic_category_entropy','semantic_query_locality','microcat_locality',
                'category_location_probability','category_location_log_support']
EXTRA_FEATURES=CONTEXT_FIELDS+[f'{kind}_{name}' for name in RANK_FIELDS for kind in ['logrank','gap']]


def encode_texts(texts,path):
    """Проверка хэша текстов защищает от неверного порядка кешированных векторов."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256('\n'.join(texts).encode()).hexdigest()
    meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        assert json.loads(meta.read_text())['texts_sha256']==digest
        return np.load(path,mmap_mode='r')
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    model=SentenceTransformer(str(ROOT/'models/multilingual-e5-small'),device=device,local_files_only=True)
    model.max_seq_length=64
    if device!='cpu':model.half()
    emb=model.encode(texts,batch_size=128,normalize_embeddings=True,show_progress_bar=True,
                     convert_to_numpy=True).astype(np.float32)
    assert np.isfinite(emb).all()
    np.save(path,emb);meta.write_text(json.dumps(dict(texts_sha256=digest,rows=len(texts))))
    del model
    if device=='mps':torch.mps.empty_cache()
    gc.collect()
    return emb


class ContextSignals:
    def __init__(self,index,history,queries,cache):
        self.index=index
        self.h=History(history,index)
        cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
        hist_text=list(self.h.query_map)
        h_emb=encode_texts(['query: '+t for t in hist_text],cache/'history_plain_e5.npy')
        # Фильтры обрабатываются отдельными признаками; здесь ищем смысл услуги.
        q_emb=encode_texts(['query: '+normalize(t) for t in queries.search_query],cache/'queries_plain_e5.npy')
        self.neighbors=np.empty((len(queries),32),dtype=np.int32)
        self.similarities=np.empty((len(queries),32),dtype=np.float32)
        for a in range(0,len(queries),128):
            with threadpool_limits(limits=4):sim=q_emb[a:a+128]@h_emb.T
            ix=np.argpartition(-sim,31,axis=1)[:,:32]
            vals=np.take_along_axis(sim,ix,axis=1)
            order=np.argsort(-vals,axis=1,kind='stable')
            self.neighbors[a:a+len(ix)]=np.take_along_axis(ix,order,axis=1)
            self.similarities[a:a+len(ix)]=np.take_along_axis(vals,order,axis=1)
        print('semantic history neighbors ready',len(queries),flush=True)
        group=history.assign(same=(history.search_location_id==history.item_location_id).astype(float))
        catstat=group.groupby('item_microcat_id').same.agg(['sum','count'])
        prior=float(group.same.mean())
        rates=(catstat['sum']+20*prior)/(catstat['count']+20)
        self.cat_locality=rates.reindex(self.h.cat_ids,fill_value=prior).to_numpy(np.float32)
        self.item_locality=rates.reindex(index.items.item_microcat_id,fill_value=prior).to_numpy(np.float32)
        trans=history.groupby(['search_location_id','item_microcat_id','item_location_id']).size()
        self.transitions={key:sub.droplevel([0,1]).to_dict() for key,sub in trans.groupby(level=[0,1])}
        self.support=history.groupby(['search_location_id','item_microcat_id']).size().to_dict()
        self.cats=index.items.item_microcat_id.to_numpy()
        self.locs=index.items.item_location_id.to_numpy()
        self.queries=queries

    def extra(self,i,ids,x):
        near=self.neighbors[i];sim=self.similarities[i]
        weights=softmax(40*(sim-sim.max()))
        p=np.asarray(weights@self.h.qcat[near]).ravel()
        mapped=self.h.corpus_cats[ids];valid=mapped>=0
        prob=np.zeros(len(ids),np.float32);prob[valid]=p[mapped[valid]]
        clicks=np.asarray(weights@self.h.qitem[near][:,ids]).ravel()
        entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum())
        locality=float(p@self.cat_locality)
        sloc=int(self.queries.iloc[i].search_location_id)
        cats=self.cats[ids];locs=self.locs[ids]
        support=np.array([self.support.get((sloc,c),0) for c in cats],np.float32)
        count=np.array([self.transitions.get((sloc,c),{}).get(l,0) for c,l in zip(cats,locs)],np.float32)
        # Backoff к обычной географии: редкий тип услуги не получает нулевую вероятность.
        locprob=x[:,FEATURES.index('location_probability')]
        conditional=(count+30*locprob)/(support+30)
        cols=[prob,np.log1p(clicks),np.full(len(ids),sim[0]),np.full(len(ids),entropy),
              np.full(len(ids),locality),self.item_locality[ids],conditional,np.log1p(support)]
        for field in RANK_FIELDS:
            score=x[:,FEATURES.index(field)]
            # Одинаковые оценки получают одинаковый ранг, независимо от item_id.
            cols.extend([np.log1p(rankdata(-score,method='average')),score-score.max()])
        out=np.column_stack(cols).astype(np.float32)
        assert out.shape==(len(ids),len(EXTRA_FEATURES)) and np.isfinite(out).all()
        return out
