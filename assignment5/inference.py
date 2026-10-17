"""HAC inference, block-bootstrap intervals, multiplicity, and chronological forecasts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def hac_ols(y,x,lags=5):
    y=np.asarray(y,float);x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    x=np.column_stack([np.ones(len(y)),x]);good=np.isfinite(y)&np.isfinite(x).all(axis=1)
    y,x=y[good],x[good];n,k=x.shape
    if n<max(30,5*k):raise ValueError(f'Too few rows for inference: n={n}, k={k}')
    if np.linalg.matrix_rank(x)<k:raise ValueError('Rank-deficient regression; constant or collinear controls')
    beta=np.linalg.lstsq(x,y,rcond=None)[0];e=y-x@beta
    scores=x*e[:,None];meat=scores.T@scores
    for lag in range(1,min(lags,n-1)+1):
        g=scores[lag:].T@scores[:-lag];meat+=(1-lag/(lags+1))*(g+g.T)
    bread=np.linalg.inv(x.T@x);cov=n/(n-k)*bread@meat@bread
    se=np.sqrt(np.maximum(np.diag(cov),0));t=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>0)
    p=2*stats.t.sf(abs(t),df=n-k)
    total=((y-y.mean())**2).sum()
    return {'n':n,'coefficients':beta.tolist(),'standard_errors':se.tolist(),'pvalues':p.tolist(),
            'ci95':np.column_stack([beta-stats.t.ppf(.975,n-k)*se,beta+stats.t.ppf(.975,n-k)*se]).tolist(),
            'r_squared':float(1-e@e/total) if total>0 else None,
            'hac_lags':lags,'condition_number':float(np.linalg.cond(x)),
            'inference':'Newey-West HAC, Bartlett weights, finite-sample n/(n-k), t approximation'}

def bh_adjust(pvalues):
    p=np.asarray(pvalues,float);q=np.full(len(p),np.nan);valid=np.isfinite(p)
    ix=np.flatnonzero(valid);order=ix[np.argsort(p[ix])];m=len(order)
    if m:q[order]=np.minimum(1,np.minimum.accumulate((p[order]*m/np.arange(1,m+1))[::-1])[::-1])
    return q

def paired_block_correlation(a,b,block=5,repeats=1000,seed=2026):
    a,b=np.asarray(a,float),np.asarray(b,float);ok=np.isfinite(a)&np.isfinite(b);a,b=a[ok],b[ok];n=len(a)
    if n<30 or np.std(a)==0 or np.std(b)==0:raise ValueError('At least 30 nonconstant pairs required')
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        if np.std(a[ix])>0 and np.std(b[ix])>0:boot.append(np.corrcoef(a[ix],b[ix])[0,1])
    return {'n':n,'pearson':float(np.corrcoef(a,b)[0,1]),'spearman':float(stats.spearmanr(a,b).statistic),
            'pearson_ci95':np.quantile(boot,[.025,.975]).tolist(),'bootstrap':'paired circular moving blocks',
            'block_length':block,'repeats':len(boot),'warning':'CI describes association; no causal interpretation'}

def walk_forward(frame,target,baseline_features,extra_features,min_train=60):
    cols=[target]+list(dict.fromkeys(baseline_features+extra_features))
    f=frame[cols].replace([np.inf,-np.inf],np.nan).dropna().sort_index()
    if len(f)<min_train+20:raise ValueError('Need at least 20 out-of-sample predictions after warmup')
    rows=[]
    for i in range(min_train,len(f)):
        train=f.iloc[:i];test=f.iloc[i:i+1];r={'date':str(f.index[i]),'actual':float(test[target].iloc[0])}
        for name,features in [('baseline',baseline_features),('augmented',baseline_features+extra_features)]:
            x=train[features].to_numpy(float);v=test[features].to_numpy(float)
            mean=x.mean(axis=0);sd=x.std(axis=0);sd[sd==0]=1
            x=np.column_stack([np.ones(len(x)),(x-mean)/sd]);v=np.column_stack([np.ones(1),(v-mean)/sd])
            # Tiny fixed ridge stabilizer, excluding intercept; not optimized on held-out data.
            penalty=np.eye(x.shape[1])*1e-6;penalty[0,0]=0
            beta=np.linalg.solve(x.T@x+penalty,x.T@train[target].to_numpy())
            r[name]=float((v@beta)[0])
        rows.append(r)
    pred=pd.DataFrame(rows);loss={name:float(np.mean((pred.actual-pred[name])**2)) for name in ['baseline','augmented']}
    diff=(pred.actual-pred.baseline)**2-(pred.actual-pred.augmented)**2
    # A descriptive improvement is not a validated trading signal.
    return pred,{'oos_n':len(pred),'mse':loss,'mse_improvement_fraction':float(1-loss['augmented']/loss['baseline']) if loss['baseline'] else None,
                 'mean_paired_squared_error_improvement':float(diff.mean()),
                 'validation':'expanding window, chronological, training-only feature standardization'}
