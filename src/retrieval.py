"""Индексы и признаки гибридного поиска. Все вычисления выполняются локально."""
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import re, time, math, gc
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
STEM = SnowballStemmer('russian')
TOKEN = re.compile(r'[a-zа-я0-9]+')
STOP = set('и в во на с со к ко у а по для из от до за под при или не о об это как'.split())

def normalize(text):
    return ' '.join(TOKEN.findall(str(text or '').lower().replace('ё','е')))

@lru_cache(maxsize=400000)
def stem(word):
    return STEM.stem(word) if len(word)>3 and re.search('[а-я]',word) else word

def tokenize(text):
    return ' '.join(stem(w) for w in normalize(text).split() if w not in STOP)

def topk(scores, k):
    """Детерминированный top-k; row index используется только для разрешения равенств."""
    k=min(k,len(scores))
    a=np.argpartition(-scores,k-1)[:k]
    return a[np.lexsort((a,-scores[a]))]

class TextIndex:
    @classmethod
    def build(cls):
        self=cls()
        self.items=pd.read_parquet(ROOT/'data/benchmark_items.parquet').fillna({'item_description_raw':''})
        self.ids=self.items.item_id.to_numpy()
        print('normalize corpus',flush=True)
        self.titles=[tokenize(x) for x in self.items.item_title_raw]
        descriptions=[tokenize(str(d)[:6500]+' '+str(p)) for d,p in zip(self.items.item_description_raw,self.items.item_infm_params_text)]
        self.raw_titles=[normalize(x) for x in self.items.item_title_raw]
        texts=[t+' '+t+' '+t+' '+d for t,d in zip(self.titles,descriptions)]
        print('word BM25 index',flush=True)
        self.words=CountVectorizer(min_df=2,max_features=220000,dtype=np.float32,token_pattern=r'(?u)\b\w+\b')
        counts=self.words.fit_transform(texts)
        lengths=np.asarray(counts.sum(axis=1)).ravel()
        df=np.bincount(counts.indices,minlength=counts.shape[1])
        self.idf=np.log(1+(len(texts)-df+.5)/(df+.5)).astype(np.float32)
        denom=1.4*(.25+.75*lengths/lengths.mean())
        counts.data=counts.data*2.4/(counts.data+np.repeat(denom,np.diff(counts.indptr)))
        counts.data*=self.idf[counts.indices]
        self.bm25=counts.tocsc()
        print('title word index',flush=True)
        self.title_vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=220000,
                                            sublinear_tf=True,dtype=np.float32,token_pattern=r'(?u)\b\w+\b')
        self.title_matrix=self.title_vectorizer.fit_transform(self.titles).tocsc()
        print('title character index',flush=True)
        self.char_vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=3,
                                            max_features=180000,sublinear_tf=True,dtype=np.float32)
        self.char_matrix=self.char_vectorizer.fit_transform(self.raw_titles).tocsc()
        print('parameter index',flush=True)
        self.params_vectorizer=TfidfVectorizer(min_df=2,sublinear_tf=True,dtype=np.float32)
        self.params_matrix=self.params_vectorizer.fit_transform(self.items.item_infm_params_text.map(tokenize)).tocsc()
        # Объект с текстами нужен для лёгких признаков покрытия слов.
        self.title_sets=[set(x.split()) for x in self.titles]
        self.items=self.items.drop(columns=['item_description_raw','item_infm_params_text'])
        del texts, descriptions, counts
        gc.collect()
        joblib.dump(self,ROOT/'artifacts/text_index.joblib',compress=0)
        print('text index saved',flush=True)
        return self

    def scores(self, row):
        q=tokenize(row.search_query)
        raw=normalize(row.search_query)
        v=self.words.transform([q]);v.data[:]=1
        bm=(self.bm25@v.T).toarray().ravel()
        title=(self.title_matrix@self.title_vectorizer.transform([q]).T).toarray().ravel()
        char=(self.char_matrix@self.char_vectorizer.transform([raw]).T).toarray().ravel()
        param=(self.params_matrix@self.params_vectorizer.transform([tokenize(row.search_infm_params_text)]).T).toarray().ravel()
        return bm,title,char,param

class History:
    def __init__(self,history, index):
        self.index=index
        self.history=history
        self.locs=index.items.item_location_id.to_numpy()
        self.cats=index.items.item_microcat_id.to_numpy()
        self.cat_ids=np.sort(history.item_microcat_id.unique())
        self.cat_map={c:j for j,c in enumerate(self.cat_ids)}
        self.corpus_cats=np.array([self.cat_map.get(c,-1) for c in self.cats])
        self.item_map={x:j for j,x in enumerate(index.ids)}
        history=history.copy()
        history['norm']=history.search_query.map(normalize)
        queries=history.norm.unique()
        self.query_map={x:j for j,x in enumerate(queries)}
        self.qvec=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1,max_features=140000,dtype=np.float32,sublinear_tf=True)
        print('history query index',len(queries),flush=True)
        self.qmat=self.qvec.fit_transform(queries).tocsc()
        qi=history.norm.map(self.query_map).to_numpy()
        ci=history.item_microcat_id.map(self.cat_map).to_numpy()
        self.qcat=sparse.coo_matrix((np.ones(len(qi),dtype=np.float32),(qi,ci)),shape=(len(queries),len(self.cat_ids))).tocsr()
        self.qfreq=np.asarray(self.qcat.sum(axis=1)).ravel()
        self.qcat=sparse.diags(1/np.maximum(self.qfreq,1))@self.qcat
        in_corpus=history.item_id.isin(self.item_map)
        self.qitem=sparse.coo_matrix((np.ones(in_corpus.sum(),dtype=np.float32),
                   (qi[in_corpus],history.loc[in_corpus,'item_id'].map(self.item_map))),shape=(len(queries),len(index.ids))).tocsr()
        self.pop=history.item_id.value_counts().reindex(index.ids,fill_value=0).to_numpy(dtype=np.float32)
        self.exact_context={}
        click_history=history.loc[in_corpus,['norm','search_location_id','item_id']].copy()
        # Преобразуем ID один раз: повторный Series.map большим словарём
        # внутри каждой группы создавал бы индекс корпуса тысячи раз.
        click_history['corpus_index']=click_history.item_id.map(self.item_map)
        for key,part in click_history.groupby(['norm','search_location_id']):
            self.exact_context[key]=(part.corpus_index.to_numpy(),np.ones(len(part),dtype=np.float32))
        trans=history.groupby(['search_location_id','item_location_id']).size()
        self.transitions={int(k):v.droplevel(0).to_dict() for k,v in trans.groupby(level=0)}
        self.location_counts=history.search_location_id.value_counts().to_dict()
        coords=pd.concat([index.items[['item_location_id','item_latitude','item_longitude']],
                          history[['item_location_id','item_latitude','item_longitude']]])
        for c in ['item_latitude','item_longitude']:coords[c]=pd.to_numeric(coords[c],errors='coerce')
        self.centers=coords.groupby('item_location_id')[['item_latitude','item_longitude']].median().to_dict('index')
        self.lat=pd.to_numeric(index.items.item_latitude,errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        self.lon=pd.to_numeric(index.items.item_longitude,errors='coerce').fillna(0).to_numpy(dtype=np.float32)
        self.geo_cache={}
        self.match_rate=history.assign(same=(history.search_location_id==history.item_location_id).astype(float)).groupby('norm').same.mean().reindex(queries).to_numpy(dtype=np.float32)

    def geo(self,loc):
        if loc not in self.geo_cache:
            trans=self.transitions.get(loc,{})
            total=self.location_counts.get(loc,0)
            unique,inv=np.unique(self.locs,return_inverse=True)
            probs=np.array([(trans.get(int(l),0)+2*(l==loc))/(total+2) for l in unique],dtype=np.float32)[inv]
            center=self.centers.get(loc)
            if center is None and trans:
                weights=[(c,self.centers[l]) for l,c in trans.items() if l in self.centers]
                if weights:
                    center={k:sum(c*v[k] for c,v in weights)/sum(c for c,v in weights) for k in ['item_latitude','item_longitude']}
            if center:
                # Haversine: километры, с корректной обработкой отсутствующих координат.
                lat1=np.radians(center['item_latitude']);lon1=np.radians(center['item_longitude'])
                lat2=np.radians(self.lat);lon2=np.radians(self.lon)
                a=np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
                dist=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
                dist[(self.lat==0)&(self.lon==0)]=10000
            else:dist=np.full(len(self.locs),10000,dtype=np.float32)
            self.geo_cache[loc]=((self.locs==loc).astype(np.float32),probs.astype(np.float32),dist.astype(np.float32))
        return self.geo_cache[loc]

    def query(self,row):
        sims=(self.qmat@self.qvec.transform([normalize(row.search_query)]).T).toarray().ravel()
        near=topk(sims,30)
        weights=np.maximum(sims[near],0)**6
        if weights.sum()>0: weights/=weights.sum()
        cat=np.asarray(weights@self.qcat[near]).ravel()
        cp=np.zeros(len(self.locs),dtype=np.float32)
        valid=self.corpus_cats>=0
        cp[valid]=cat[self.corpus_cats[valid]]
        clicks=np.asarray(weights@self.qitem[near]).ravel().astype(np.float32)
        exact=np.zeros(len(self.locs),dtype=np.float32)
        ec=self.exact_context.get((normalize(row.search_query),row.search_location_id))
        if ec is not None:np.add.at(exact,ec[0],ec[1])
        return cp,clicks,exact,float(sims[near[0]]),float(weights@self.match_rate[near])

FEATURES=['bm25','bm25_relative','title_cos','char_cos','params_cos','dense_cos','same_location',
          'location_probability','log_distance','near_10km','near_50km','cat_probability','history_clicks',
          'context_clicks','history_max_sim','query_locality','popularity','rating','reviews','log_price',
          'phone_hidden','message_forbidden','title_coverage','description_coverage','exact_phrase',
          'query_words','title_words','params_present','rating_filter','rating_satisfies',
          'bm25_geo','dense_geo','char_geo','category_id_match']

class Retriever:
    def __init__(self,index,history,dense_items=None,profile='current'):
        self.index=index
        self.history=History(history,index)
        self.dense_items=dense_items
        self.profile=profile
        items=index.items
        self.static=np.column_stack([
            self.history.pop,items.item_rating.fillna(-1),np.log1p(items.item_rating_reviews_count.fillna(0)),
            np.log1p(pd.to_numeric(items.item_price,errors='coerce').fillna(0).clip(lower=0)),
            items.item_is_phone_hidden.astype(float),items.item_is_message_forbidden.astype(float)]).astype(np.float32)

    def retrieve(self,row,dense_query=None,*,extra_candidates=None):
        bm,title,char,param=self.index.scores(row)
        same,prob,dist=self.history.geo(int(row.search_location_id))
        cat,clicks,exact,near_sim,locality=self.history.query(row)
        dense=np.asarray(self.dense_items@dense_query,dtype=np.float32) if dense_query is not None else np.zeros(len(bm),dtype=np.float32)
        bmrel=bm/max(bm.max(),1e-6)
        # География — только prior. В broad-профиле её вес снижен, чтобы не
        # терять исполнителей из соседних городов и дистанционные услуги.
        if self.profile == 'broad':
            geo=.35*same+.15*np.sqrt(prob)+.05*np.exp(-dist/35)
        else:
            geo=.65*same+.25*np.sqrt(prob)+.10*np.exp(-dist/35)
        bmgeo=bmrel*(.30+.70*geo)
        charg=char*(.30+.70*geo)
        denseg=dense+.10*geo
        if self.profile in {'broad', 'wide'}:
            lists=[topk(bmgeo,420),topk(charg,360),topk(title*(.3+.7*geo),320),
                   topk(bmrel,180),topk(char,160),
                   topk((.7*bmrel+.3*char)*(.25+.75*cat)*(.3+.7*geo),260),
                   topk(clicks*(.2+.8*geo)+2*exact,120)]
        else:
            lists=[topk(bmgeo,170),topk(charg,120),topk(title*(.3+.7*geo),100),topk(bmrel,45),
                   topk(char,30),topk((.7*bmrel+.3*char)*(.25+.75*cat)*(.3+.7*geo),70),
                   topk(clicks*(.2+.8*geo)+2*exact,40)]
        if dense_query is not None:
            lists.extend([topk(denseg,420 if self.profile in {'broad', 'wide'} else 180),
                          topk(dense,180 if self.profile in {'broad', 'wide'} else 60)])
        # Дополнительный канал поиска; исходный пул сохраняется.
        # Вызывающий код передаёт индексы строк, полученные без меток релевантности.
        if extra_candidates is not None:
            lists.append(np.asarray(extra_candidates,dtype=np.int64))
        candidates=np.unique(np.concatenate(lists))
        # Не добавляем положительные объекты искусственно: качество включает промахи retrieval.
        c=candidates
        tokens=set(tokenize(row.search_query).split())
        nt=max(len(tokens),1)
        coverage=np.array([len(tokens&self.index.title_sets[j])/nt for j in c],dtype=np.float32)
        qv=self.index.words.transform([tokenize(row.search_query)])
        desc_cov=np.asarray((self.index.bm25[:,qv.indices]>0).sum(axis=1)).ravel()[c]/nt
        phrase=normalize(row.search_query)
        rating_filter='4 звезды' in row.search_infm_params_text
        features=np.column_stack([
            bm[c],bmrel[c],title[c],char[c],param[c],dense[c],same[c],prob[c],np.log1p(dist[c]),
            (dist[c]<10),(dist[c]<50),cat[c],np.log1p(clicks[c]),np.log1p(exact[c]),
            np.full(len(c),near_sim),np.full(len(c),locality),self.static[c],
            coverage,desc_cov,[float(phrase in self.index.raw_titles[j]) for j in c],
            np.full(len(c),nt),[len(self.index.title_sets[j]) for j in c],
            np.full(len(c),bool(row.search_infm_params_text)),np.full(len(c),rating_filter),
            (not rating_filter)|(self.static[c,1]>=4),bmgeo[c],denseg[c],charg[c],
            (self.index.items.item_category_id.to_numpy()[c]==row.search_category)
        ]).astype(np.float32)
        assert features.shape[1]==len(FEATURES)
        baseline=.5*bmgeo[c]+.3*charg[c]+.2*(title[c]*(.3+.7*geo[c]))
        return c,features,baseline

if __name__=='__main__':
    from retrieval import TextIndex
    TextIndex.build()
