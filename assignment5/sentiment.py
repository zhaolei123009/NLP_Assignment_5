"""Reproducible VADER topic-affect estimation using only pre-cutoff text.

This estimator measures expressed affect in keyword-selected discussion, not
verified voter opinion or partisan advantage. No independent domain accuracy
claim is made. Never train or optimize the lexicon using financial outcomes.
"""
from __future__ import annotations
import json, pathlib, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=pathlib.Path(__file__).resolve().parent
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
START='2026-04-07T00:00:00Z'
ASOF='2026-10-07T23:04:38Z'
from equities import asof_values
from inference import hac_ols, paired_block_correlation, bh_adjust, walk_forward
from topics import TopicDiscovery

PATTERNS={
 'economy':r'\b(?:econom\w*|inflation|affordab\w*|prices?|costs?|wages?|jobs?|tariffs?|tax(?:es)?|housing|rent)\b',
 'governance':r'\b(?:democracy|democratic (?:norms|values)|corrupt\w*|constitution\w*|authoritarian\w*|dictator\w*|ethics|accountability|elections?|voting|votes?|voters?)\b',
 'immigration':r'\b(?:immigra\w*|deport\w*|border|ice|asylum|migrants?)\b',
 'crime':r'\b(?:crimes?|criminal\w*|police|policing|violence|violent|murder\w*|safety|prison\w*|jail\w*)\b',
}
TOPICS=['pooled','election_all']+list(PATTERNS)

def check_cutoff(documents):
    f=documents.copy()
    f['available_at']=pd.to_datetime(f.available_at,utc=True)
    if not f.available_at.between(pd.Timestamp(START),pd.Timestamp(ASOF)).all():
        raise ValueError('Document timestamp outside original study window')
    if f.document_id.duplicated().any():raise ValueError('Duplicate document IDs')
    return f

def score_documents(documents):
    f=check_cutoff(documents);analyzer=SentimentIntensityAnalyzer();rows=[]
    seen=set()
    for r in f.sort_values(['available_at','document_id']).itertuples():
        text=str(r.text);normalized=re.sub(r'\W+',' ',text.lower()).strip()
        if normalized in seen:continue
        seen.add(normalized)
        sentences=[x.strip() for x in re.split(r'(?<=[.!?])\s+',text) if x.strip()]
        matched={topic:' '.join(x for x in sentences if re.search(pattern,x,re.I))
                 for topic,pattern in PATTERNS.items()}
        fragments={'election_all':text}
        fragments.update({k:v for k,v in matched.items() if v})
        # Each original sentence occurs once in the pooled issue-affect score.
        pooled=' '.join(x for x in sentences if any(re.search(pat,x,re.I) for pat in PATTERNS.values()))
        if pooled:fragments['pooled']=pooled
        for topic,fragment in fragments.items():
            score=analyzer.polarity_scores(fragment)
            rows.append({'document_id':r.document_id,'available_at':r.available_at,'source':r.source,
                         'topic':topic,'author_id':str(getattr(r,'author_id','') or r.document_id),
                         'event_id':str(getattr(r,'event_id','') or r.document_id),
                         'compound':score['compound'],'positive':score['pos'],'neutral':score['neu'],'negative':score['neg'],
                         'scored_words':len(fragment.split()),'url':r.url})
    return pd.DataFrame(rows)

def aggregate_daily(scored):
    """Equal-author daily means; empty days stay missing, never interpolated."""
    first=pd.Timestamp(START).floor('D');last=pd.Timestamp(ASOF).floor('D')
    grid=pd.date_range(first,last,freq='D')
    s=scored.copy();s['day']=pd.to_datetime(s.available_at,utc=True).dt.floor('D')
    rows=[]
    for source,gsource in s.groupby('source'):
        for topic in TOPICS:
            g=gsource[gsource.topic==topic]
            for day in grid:
                d=g[g.day==day]
                # Do not treat a partial final day as a completed daily signal.
                available=day+pd.Timedelta(days=1)
                authors=d.groupby('author_id').compound.mean()
                stories=d.groupby('event_id').compound.mean()
                valid=available<=pd.Timestamp(ASOF) and len(d)>0
                rows.append({'source':source,'topic':topic,'day':day,'available_at':available,
                             'sentiment':float(authors.mean()) if valid else np.nan,
                             'story_balanced_sentiment':float(stories.mean()) if valid else np.nan,
                             'comment_count':len(d),'authors':len(authors),'stories':len(stories),
                             'largest_story_share':float(d.event_id.value_counts(normalize=True).max()) if len(d) else np.nan,
                             'low_coverage':len(authors)<3 or len(stories)<2,'completed_bucket':available<=pd.Timestamp(ASOF)})
    return pd.DataFrame(rows)

def relation(frame,y,features):
    f=frame[[y]+features].replace([np.inf,-np.inf],np.nan).dropna()
    result={'n':len(f),'pearson':None,'spearman':None,'pvalue':None,'coefficient':None,
            'ci95_low':None,'ci95_high':None,'inference':'descriptive only'}
    if len(f)>=3 and f[y].std()>0 and f[features[0]].std()>0:
        result['pearson']=float(f[y].corr(f[features[0]]))
        result['spearman']=float(f[y].corr(f[features[0]],method='spearman'))
    if len(f)>=30:
        try:
            reg=hac_ols(f[y],f[features],lags=5)
            corr=paired_block_correlation(f[y],f[features[0]],block=5,repeats=1000)
            result.update(pvalue=reg['pvalues'][1],coefficient=reg['coefficients'][1],
                          ci95_low=corr['pearson_ci95'][0],ci95_high=corr['pearson_ci95'][1],
                          inference='OLS slope, Newey-West HAC(5); paired block(5) Pearson interval',
                          regression=reg)
        except ValueError as e:result['inference']='descriptive only: '+str(e)
    return result

def odds_panel(series,market,column='sentiment'):
    f=series.set_index('available_at').sort_index().copy()
    f.index=pd.to_datetime(f.index,utc=True)
    # UTC bucket end marks availability; no use of the incomplete final day.
    f=f.loc[f.index<=pd.Timestamp(ASOF)]
    q=asof_values(market,f.index,tolerance='24h')
    f['probability']=q.probability
    logit=np.log(f.probability.clip(.0001,.9999)/(1-f.probability.clip(.0001,.9999)))
    f['odds_change']=logit.diff()
    f['odds_next_change']=f.odds_change.shift(-1)
    f['sentiment_signal']=f[column]
    f['sentiment_change']=f[column].diff()
    return f

def basket_panel(series,basket,column='sentiment',previous_close=True):
    f=basket.copy().sort_index()
    closes=(f.index+pd.Timedelta(hours=16)).tz_localize('America/New_York').tz_convert('UTC')
    cutoffs=pd.Series(closes).shift(1) if previous_close else pd.Series(closes)
    left=pd.DataFrame({'date':f.index,'signal_cutoff':list(cutoffs)}).dropna()
    right=series[['available_at',column,'comment_count','authors','stories','low_coverage']].copy()
    right['available_at']=pd.to_datetime(right.available_at,utc=True)
    right=right.loc[right.available_at<=pd.Timestamp(ASOF)].sort_values('available_at')
    # Match calendar rows including missing signals; never skip a missing day.
    m=pd.merge_asof(left.sort_values('signal_cutoff'),right,left_on='signal_cutoff',right_on='available_at',
                    direction='backward',tolerance=pd.Timedelta('26h')).set_index('date')
    f=f.join(m)
    f['sentiment_signal']=f[column]
    f['lag_basket']=f.equal_weight_basket_ar.shift(1)
    f['lag_spread']=f.pro_democratic_spread_ar.shift(1)
    f['lag_spy']=f.spy_return.shift(1)
    valid=f.available_at.notna()&f.signal_cutoff.notna()
    if (f.loc[valid,'available_at']>f.loc[valid,'signal_cutoff']).any():raise AssertionError('Look-ahead in stock signal')
    return f

def main(news,baskets):
    out=ROOT/'results';out.mkdir(exist_ok=True)
    social=pd.read_csv(ROOT/'data/social.csv').fillna('')
    model=TopicDiscovery().fit(social,'2026-05-31T23:59:59Z',n_topics=4)
    assignments=model.transform(social)
    description=model.description(n_terms=7)
    description.update(documents=len(social),heldout_documents=len(social)-model.train_n,
                       descriptive_topic_counts=assignments.drop(columns='document_id').idxmax(axis=1).value_counts().to_dict())
    news['author_id']=news.publisher;news['event_id']=news.event_id.replace('',np.nan).fillna(news.document_id)
    scores=pd.concat([score_documents(social),score_documents(news)],ignore_index=True)
    scores.to_csv(out/'document_scores.csv',index=False)
    daily=aggregate_daily(scores);daily.to_csv(out/'daily_indices.csv',index=False)
    market=pd.read_csv(ROOT/'data/market.csv')
    rows=[];predictions=[];oos={};panel_audit=[];forecast_frames=[];primary_odds=None;primary_stock=None
    for source in ['social','news']:
      for topic in TOPICS:
        series=daily[(daily.source==source)&(daily.topic==topic)]
        f=odds_panel(series,market)
        if source=='social' and topic=='pooled':primary_odds=f
        specs=[('odds_same_day','odds_change',['sentiment_signal']),
               ('odds_next_day','odds_next_change',['sentiment_signal','odds_change']),
               ('odds_sentiment_changes','odds_change',['sentiment_change'])]
        for label,target,features in specs:
            result=relation(f,target,features)
            rows.append({'source':source,'topic':topic,'specification':'market_adjusted','test':label,**result})
        for adjustment in ['market_adjusted','factor_adjusted']:
            basket=baskets[adjustment]
            stock=basket_panel(series,basket)

            for label,target,lag in [('equal_basket_next_session','equal_weight_basket_ar','lag_basket'),
                                     ('directional_spread_next_session','pro_democratic_spread_ar','lag_spread')]:
                result=relation(stock,target,['sentiment_signal',lag,'lag_spy'])
                rows.append({'source':source,'topic':topic,'specification':adjustment,'test':label,**result})
            if source=='social' and topic=='pooled' and adjustment=='market_adjusted':
                primary_stock=stock
                for target,lag in [('equal_weight_basket_ar','lag_basket'),('pro_democratic_spread_ar','lag_spread')]:
                    try:
                        pred,summary=walk_forward(stock,target,[lag,'lag_spy'],['sentiment_signal'],min_train=40)
                        pred['target']=target;forecast_frames.append(pred);oos[target]=summary
                    except ValueError as e:oos[target]={'scope':'descriptive association only','reason':str(e)}
        if source=='social' and topic=='pooled':
            try:
                pred,summary=walk_forward(f,'odds_next_change',['odds_change'],['sentiment_signal'],min_train=40)
                pred['target']='odds';forecast_frames.append(pred);oos['odds']=summary
            except ValueError as e:oos['odds']={'scope':'descriptive association only','reason':str(e)}
            # Fixed coverage and thread-weighting sensitivities, not outcome-selected.
            for name,col in [('author_balanced','sentiment'),('story_balanced','story_balanced_sentiment'),('higher_coverage','sentiment')]:
                alt=series.copy()
                if name=='higher_coverage':alt.loc[alt.low_coverage,col]=np.nan
                af=odds_panel(alt,market,column=col)
                ar=relation(af,'odds_change',['sentiment_signal'])
                panel_audit.append({'sensitivity':name,'outcome':'same_day_odds',**ar})
    results=pd.DataFrame(rows)
    # All 84 configured contrasts plus three coverage checks form one family;
    # non-estimable tests have p=1
    # only for multiplicity accounting, never a fabricated estimated p-value.
    sensitivity=pd.DataFrame(panel_audit)
    allp=pd.concat([results.pvalue,sensitivity.pvalue],ignore_index=True).fillna(1.0)
    allq=bh_adjust(allp)
    results['qvalue_bh']=allq[:len(results)]
    results.loc[results.pvalue.isna(),'qvalue_bh']=np.nan
    clean=results.drop(columns=['regression'],errors='ignore')
    clean.to_csv(out/'sentiment_estimates.csv',index=False)
    daily['covered_low_day']=daily.low_coverage&daily.sentiment.notna()
    coverage=daily.groupby(['source','topic']).agg(days=('sentiment','count'),comments=('comment_count','sum'),
                  low_coverage_days=('covered_low_day','sum')).reset_index()
    coverage.to_csv(out/'coverage.csv',index=False)
    sensitivity['qvalue_bh']=allq[len(results):]
    sensitivity.loc[sensitivity.pvalue.isna(),'qvalue_bh']=np.nan
    sensitivity.drop(columns=['regression'],errors='ignore').to_csv(out/'coverage_sensitivity.csv',index=False)
    summary={'cutoff':ASOF,'start':START,'method':'VADER 3.3.2 issue-sentence affect; equal-author daily aggregation',
             'social_documents':len(social),'news_documents':len(news),'scored_rows':len(scores),'social_topics':description,
             'tests_in_family':len(allp),'estimated_hac_tests':int(results.pvalue.notna().sum()+sensitivity.pvalue.notna().sum()),
             'significant_bh_005':int((results.qvalue_bh<.05).sum()+(sensitivity.qvalue_bh<.05).sum()),'oos':oos,
             'limitations':['Hacker News users are not a representative voter sample.',
                            'General affect is not sentiment toward a party or validated partisan advantage.',
                            'Sparse issue-day samples, especially immigration, limit inference.',
                            'Historical text can have been edited after creation; collection is not an immutable point-in-time archive.',
                            'No independent domain-specific labeled accuracy estimate or complete poll/macro controls.']}
    pd.concat(forecast_frames,ignore_index=True).to_csv(out/'forecasts.csv',index=False)
    primary_odds.to_csv(out/'odds_panel.csv',index_label='cutoff')
    primary_stock.to_csv(out/'basket_panel.csv',index_label='date')
    # A regular calendar plot retains all missing days.
    g=daily[(daily.source=='social')&(daily.topic=='pooled')].copy()
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,constrained_layout=True)
    axes[0].plot(pd.to_datetime(g.day),g.sentiment,color='#21577e',marker='.',ms=2,lw=.8)
    axes[0].set_ylabel('Issue-affect index [-1, 1]');axes[0].axhline(0,color='gray',lw=.6)
    axes[0].set_title('Historical election-discussion affect | no interpolation across missing days')
    axes[1].bar(pd.to_datetime(g.day),g.comment_count,color='#7294ae',width=.8)
    axes[1].set_ylabel('Issue-matched comments');axes[1].set_xlabel('2026 (UTC calendar days)')
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(out/'sentiment_over_time.png',dpi=180);plt.close(fig)
    f=primary_odds
    b=primary_stock
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax,frame,y,label in [(axes[0],f,'odds_change','Daily logit sweep-odds change'),
                              (axes[1],b,'equal_weight_basket_ar','Next-session basket abnormal return')]:
        z=frame[['sentiment_signal',y]].dropna();ax.scatter(z.sentiment_signal,z[y],s=18,alpha=.6)
        if len(z)>2:
            line=np.polyfit(z.sentiment_signal,z[y],1);xs=np.array([z.sentiment_signal.min(),z.sentiment_signal.max()]);ax.plot(xs,np.polyval(line,xs),color='#b95a34')
        ax.set_xlabel('Pooled issue-affect index');ax.set_ylabel(label);ax.grid(alpha=.15)
    fig.suptitle('Sentiment associations | descriptive, not causal')
    fig.savefig(out/'sentiment_associations.png',dpi=180);plt.close(fig)
    return summary


