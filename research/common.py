import pandas as pd, numpy as np
from collections import defaultdict
D='public/'
P=[0,1,2,5,20]; PC=['p_0','p_1','p_2','p_5','p_20']
def load(split):
    q=pd.read_csv(D+split+'/queries.csv'); h=pd.read_csv(D+split+'/history.csv')
    try: q=q.merge(pd.read_csv(D+split+'/conditions.csv'),on='query_id')
    except FileNotFoundError: pass
    try: q=q.merge(pd.read_csv(D+split+'/labels.csv')[['query_id','outcome']],on='query_id')
    except FileNotFoundError: pass
    seqs={m:dict(zip(g.tick,g.payout.map(P.index))) for m,g in h.groupby('machine_id')}
    return q,seqs
VQ,VS=load('validation')
T=pd.read_csv(D+'validation/targets.csv').set_index('query_id')[PC]
REF=pd.read_csv(D+'validation/reference.csv').set_index('query_id')[PC]
def _err(pred,qs):
    e=((pred.loc[qs.query_id].values-T.loc[qs.query_id].values)**2).sum(1)
    return pd.Series(e,index=qs.index)
def score(pred, qs=None, verbose=False):
    """pred: DataFrame index query_id cols PC. returns points/60 and per-cond ratios"""
    qs=VQ if qs is None else qs
    pred=pred.div(pred.sum(1),axis=0)
    d=qs[['query_id','episode_id','condition']].copy()
    d['e']=_err(pred,qs); d['r']=_err(REF,qs)
    ep=d.groupby(['condition','episode_id'])[['e','r']].mean().groupby('condition').mean()
    rat=ep.e/ep.r.clip(lower=1e-9); R=rat.mean()
    if verbose: print({k:round(v,3) for k,v in rat.items()})
    return 60/(1+R), rat.to_dict()
