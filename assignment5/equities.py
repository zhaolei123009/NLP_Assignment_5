"""Joint-contract alignment and rolling market/factor-adjusted equity analysis."""
import json,pathlib
import numpy as np
import pandas as pd
from inference import hac_ols,bh_adjust,paired_block_correlation,walk_forward

ROOT=pathlib.Path(__file__).resolve().parent
START='2026-04-07T00:00:00Z'
ASOF='2026-10-07T23:04:38Z'

def asof_values(observations,targets,value='probability',tolerance='30h'):
    columns=['timestamp',value]+(['resolution_seconds'] if 'resolution_seconds' in observations else [])
    obs=observations[columns].copy();obs.timestamp=pd.to_datetime(obs.timestamp,utc=True)
    if obs.timestamp.duplicated().any():raise ValueError('Duplicate quote timestamps')
    # When an API supplies a bucket width but does not establish which edge is
    # known at the timestamp, impose a full-width availability embargo.
    obs['available_time']=obs.timestamp
    if 'resolution_seconds' in obs:
        obs['available_time']=obs.timestamp+pd.to_timedelta(obs.resolution_seconds,unit='s')
    targets=pd.DatetimeIndex(targets)
    a=pd.DataFrame({'cutoff':targets}).sort_values('cutoff')
    f=pd.merge_asof(a,obs.sort_values('available_time'),left_on='cutoff',right_on='available_time',direction='backward',tolerance=pd.Timedelta(tolerance))
    f['quote_age_hours']=(f.cutoff-f.timestamp).dt.total_seconds()/3600
    return f.set_index('cutoff')

def rolling_abnormal_returns(prices,benchmark='SPY',window=252,min_periods=126,industry=None):
    """Market model coefficients always estimated before the return being evaluated."""
    r=prices.pct_change(fill_method=None);rows=[]
    for ticker in ['FSLR','HCA','GEO']:
        factors=[benchmark]+([industry[ticker]] if industry else [])
        for i in range(min_periods+1,len(r)):
            past=r.iloc[max(0,i-window):i][[ticker]+factors].dropna()
            current=r.iloc[i]
            if len(past)<min_periods or current[[ticker]+factors].isna().any():continue
            x=np.column_stack([np.ones(len(past)),past[factors].to_numpy()])
            if np.linalg.matrix_rank(x)<x.shape[1]:continue
            beta=np.linalg.lstsq(x,past[ticker].to_numpy(),rcond=None)[0]
            expected=float(np.r_[1,current[factors].to_numpy()]@beta)
            rows.append({'date':r.index[i],'ticker':ticker,'actual_return':float(current[ticker]),
                         'expected_return':expected,'abnormal_return':float(current[ticker]-expected),
                         'alpha':float(beta[0]),'market_beta':float(beta[1]),'estimation_n':len(past),
                         'last_training_date':str(past.index.max()),'factors':'+'.join(factors)})
    return pd.DataFrame(rows)

def basket_series(abnormal):
    wide=abnormal.pivot(index='date',columns='ticker',values='abnormal_return').sort_index()
    f=pd.DataFrame(index=wide.index)
    f['equal_weight_basket_ar']=wide[['FSLR','HCA','GEO']].mean(axis=1,skipna=False)
    f['pro_democratic_spread_ar']=.5*wide.FSLR+.5*wide.HCA-wide.GEO
    f['pro_democratic_long_leg_ar']=.5*(wide.FSLR+wide.HCA)
    f['geo_leg_ar']=wide.GEO
    return f.join(wide.add_suffix('_ar'))

def equity_analysis(market):
    p=pd.read_csv(ROOT/'data/prices.csv',index_col='date',parse_dates=True).sort_index()
    if p.index.max()>pd.Timestamp(ASOF).tz_localize(None).normalize():raise ValueError('Post-cutoff prices')
    industry={'FSLR':'TAN','HCA':'XLV','GEO':'IWM'}
    outputs={};regrows=[];basketout=None;baskets={}
    for kind,ind in [('market_adjusted',None),('factor_adjusted',industry)]:
        ar=rolling_abnormal_returns(p,industry=ind)
        basket=basket_series(ar);basket=basket.loc[basket.index>=pd.Timestamp(START).tz_localize(None).normalize()]
        # Align contract quotes to 16:00 NY, explicitly recording the age of coarse prices.
        close=(basket.index+pd.Timedelta(hours=16)).tz_localize('America/New_York').tz_convert('UTC')
        q=asof_values(market,close);q.index=basket.index
        basket=basket.join(q[['probability','quote_age_hours']])
        # Difference strictly across adjacent completed stock sessions, not dropped observations.
        basket['delta_probability']=basket.probability.diff()
        basket['delta_logit_probability']=np.log(basket.probability.clip(.0001,.9999)/(1-basket.probability.clip(.0001,.9999))).diff()
        basket['lag_delta_probability']=basket.delta_probability.shift(1)
        market_return=p.SPY.pct_change(fill_method=None);basket['spy_return']=market_return.reindex(basket.index)
        baskets[kind]=basket
        if kind=='market_adjusted':basketout=basket
        outputs[kind]={}
        for target in ['FSLR_ar','HCA_ar','GEO_ar','equal_weight_basket_ar','pro_democratic_spread_ar']:
            same=basket[[target,'delta_probability']].dropna()
            predictive=basket[[target,'lag_delta_probability']].dropna()
            for label,data,feature in [('contemporaneous',same,'delta_probability'),('lagged',predictive,'lag_delta_probability')]:
                if len(data)<30:continue
                fit=hac_ols(data[target],data[[feature]],lags=5)
                corr=paired_block_correlation(data[target],data[feature])
                key=target+'_'+label;outputs[kind][key]={'regression':fit,'correlation':corr}
                regrows.append({'adjustment':kind,'asset':target,'test':label,'n':fit['n'],
                  'coefficient':fit['coefficients'][1],'se_hac':fit['standard_errors'][1],
                  'pvalue':fit['pvalues'][1],'ci_low':fit['ci95'][1][0],'ci_high':fit['ci95'][1][1],
                  'r_squared':fit['r_squared'],'corr_pearson':corr['pearson'],'corr_spearman':corr['spearman'],
                  'corr_ci_low':corr['pearson_ci95'][0],'corr_ci_high':corr['pearson_ci95'][1]})
        # Note: this assesses market-odds features only, not the absent voter sentiment feature.
        f=basket.copy();f['lag_spread_ar']=f.pro_democratic_spread_ar.shift(1);f['lag_spy_return']=f.spy_return.shift(1)
        try:
            predictions,validation=walk_forward(f,'pro_democratic_spread_ar',['lag_spread_ar','lag_spy_return'],['lag_delta_probability'],min_train=60)
            outputs[kind]['odds_oos']=validation
        except ValueError as e:outputs[kind]['odds_oos']={'unavailable':str(e)}
    table=pd.DataFrame(regrows);table['qvalue_bh']=bh_adjust(table.pvalue);table.to_csv(ROOT/'results/odds_equity.csv',index=False)
    returns=p.pct_change(fill_method=None).loc[basketout.index]
    result={'adjustments':outputs,'dates':{'start':str(basketout.index.min().date()),'end':str(basketout.index.max().date()),'sessions':len(basketout)},
            'quote_age_hours':basketout.quote_age_hours.describe().to_dict(),
            'total_returns':{t:float((1+returns[t].dropna()).prod()-1) for t in ['FSLR','HCA','GEO','SPY']},
            'cumulative_abnormal_return_sum':{k:float(basketout[k].sum()) for k in ['FSLR_ar','HCA_ar','GEO_ar','equal_weight_basket_ar','pro_democratic_spread_ar']},
            'selection_warning':'Basket selected at research date, retrospectively; not a historical preregistered trading strategy',
            'factor_warning':'GEO uses IWM as a size robustness control, not a detention-industry benchmark',
            'price_warning':'Sparse vendor buckets matched backward at close; v2 observations have full-width availability embargo. No historical bid-ask, volume or executable-price verification',
            'risk_free_warning':'Market-model abnormal returns; risk-free rate not supplied, not CAPM excess returns'}
    return result,baskets
