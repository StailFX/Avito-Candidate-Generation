"""Линейная адаптация query E5 к пространству объявлений.

Ridge-регрессия учит сдвиг относительно исходного преобразования I. Один
нормализованный запрос имеет один вес независимо от числа взаимодействий.
История передаётся уже очищенной от оценочных контекстов. Модель E5 заморожена.
"""
import numpy as np
from scipy.stats import rankdata

ADAPTER_FEATURES=['plain_e5_cos','adapted_e5_cos','adapted_logrank','adapted_gap']

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def fit_mapping(x,y,alpha=10.):
    """Минимизирует ||X W - Y||² + alpha ||W - I||²."""
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert x.shape==y.shape
    eye=np.eye(x.shape[1])
    return np.linalg.solve(x.T@x+alpha*eye,x.T@y+alpha*eye).astype(np.float32)

def adapter_features(query, adapted, docs):
    base=docs@query;score=docs@adapted
    return np.column_stack([base,score,np.log1p(rankdata(-score,method='average')),
                            score-score.max()]).astype(np.float32)
