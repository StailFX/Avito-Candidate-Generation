"""Строгая проверка формата; идентификаторы никогда не преобразуются в числа."""
import argparse,json,re
from pathlib import Path
import pandas as pd


def validate(path,data):
    answer=pd.read_csv(path,dtype=str,keep_default_na=False,encoding='utf-8')
    queries=pd.read_parquet(Path(data)/'benchmark_queries.parquet',columns=['query_id'])
    items=pd.read_parquet(Path(data)/'benchmark_items.parquet',columns=['item_id'])
    assert answer.columns.tolist()==['query_id','answer'], 'Неверные колонки'
    assert answer.query_id.is_unique, 'Повтор query_id'
    assert set(answer.query_id)==set(queries.query_id), 'Пропущенные или лишние query_id'
    assert answer.query_id.str.len().eq(16).all(), 'Неверная длина query_id'
    allowed=set(items.item_id)
    lengths=[]
    for row in answer.itertuples(index=False):
        ids=row.answer.split(' ') if row.answer else []
        assert len(ids)<=50, f'Больше 50 кандидатов: {row.query_id}'
        assert len(ids)==len(set(ids)), f'Дубликат item_id: {row.query_id}'
        assert all(re.fullmatch('[0-9a-f]{16}',s) for s in ids), f'Неверный item_id: {row.query_id}'
        assert set(ids)<=allowed, f'Неизвестный item_id: {row.query_id}'
        lengths.append(len(ids))
    report={'valid':True,'rows':len(answer),'min_candidates':min(lengths),'max_candidates':max(lengths),'columns':answer.columns.tolist()}
    Path(path).with_suffix('.validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('answer');p.add_argument('--data',default=str(Path(__file__).resolve().parents[1]/'data'));a=p.parse_args()
    print(json.dumps(validate(a.answer,a.data),ensure_ascii=False,indent=2))
