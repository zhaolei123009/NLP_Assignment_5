"""Run all grading-relevant analysis offline: python main.py."""
import datetime,json,pathlib
import pandas as pd
from equities import equity_analysis,ASOF,START
from topics import discover_news
from sentiment import main as estimate_sentiment

ROOT=pathlib.Path(__file__).resolve().parent

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    market=pd.read_csv(ROOT/'data/market.csv')
    times=pd.to_datetime(market.timestamp,utc=True)
    if not times.between(pd.Timestamp(START),pd.Timestamp(ASOF)).all():raise ValueError('Market timestamp outside historical window')
    if not market.probability.between(0,1).all():raise ValueError('Invalid contract probability')
    equities,baskets=equity_analysis(market)
    news,topics=discover_news(pd.read_csv(ROOT/'data/news.csv'))
    (ROOT/'results/news_topics.json').write_text(json.dumps(topics,indent=2))
    sentiment=estimate_sentiment(news,baskets)
    result={'historical_cutoff':ASOF,'executed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'equities':equities,'sentiment':sentiment}
    (ROOT/'results/summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    table=pd.read_csv(ROOT/'results/sentiment_estimates.csv')
    key=table[(table.source=='social')&(table.topic=='pooled')&(table.specification=='market_adjusted')&
              (table.test!='odds_sentiment_changes')][['test','n','pearson','pvalue','qvalue_bh','ci95_low','ci95_high']]
    key.to_csv(ROOT/'results/key_results.csv',index=False)
    print(key.to_string(index=False))
    print('Completed offline analysis. Saved outputs are in results/.')

if __name__=='__main__':main()
