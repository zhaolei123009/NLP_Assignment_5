"""Optional read-only API collection; main.py uses frozen CSVs instead.

New downloads go to acquired/, never overwrite the grading reference inputs.
"""
import datetime as dt,hashlib,html,json,pathlib,re,urllib.request,urllib.parse
import pandas as pd

ROOT=pathlib.Path(__file__).resolve().parent
START=pd.Timestamp('2026-04-07T00:00:00Z')
CUTOFF=pd.Timestamp('2026-10-07T23:04:38Z')
ELECTORAL=re.compile(r'\b(?:trump|democrats?|republicans?|congress(?:ional)?|senat(?:e|or|ors)|biden|harris|maga|gop|elections?|votes?|voters?|voting|presiden\w*|races)\b',re.I)

def request_json(name,url,params):
    url+='?'+urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Assignment5 academic research'}),timeout=35) as response:
        raw=response.read()
    folder=ROOT/'acquired';folder.mkdir(exist_ok=True)
    (folder/(name+'.json')).write_bytes(raw)
    log=folder/'requests.json';rows=json.loads(log.read_text()) if log.exists() else []
    rows.append({'url':url,'retrieved_at':dt.datetime.now(dt.timezone.utc).isoformat(),'sha256':hashlib.sha256(raw).hexdigest()})
    log.write_text(json.dumps(rows,indent=2))
    return json.loads(raw)

def fetch_news():
    obj=request_json('gdelt','https://api.gdeltproject.org/api/v2/doc/doc',
        {'query':'(midterm OR midterms) sourcelang:english sourcecountry:US','mode':'artlist','format':'json',
         'startdatetime':START.strftime('%Y%m%d%H%M%S'),'enddatetime':CUTOFF.strftime('%Y%m%d%H%M%S'),
         'maxrecords':250,'sort':'DateDesc'})
    print('News API returned',len(obj.get('articles',[])),'articles; this is a capped sample.')

def fetch_social():
    boundaries=pd.date_range('2026-04-01','2026-11-01',freq='MS',tz='UTC');rows={}
    for query in ['midterm','midterms']:
      for lo,hi in zip(boundaries,boundaries[1:]):
        start=max(START,lo);end=min(CUTOFF,hi-pd.Timedelta(seconds=1))
        if end<start:continue
        params={'query':query,'tags':'comment','numericFilters':f'created_at_i>={int(start.timestamp())},created_at_i<={int(end.timestamp())}',
                'hitsPerPage':100,'typoTolerance':'false','removeWordsIfNoResults':'none'}
        page=0
        while True:
            obj=request_json(f'hn_{query}_{lo:%Y%m}_{page}','https://hn.algolia.com/api/v1/search_by_date',dict(params,page=page))
            if obj.get('nbHits',0)>=1000:raise ValueError('Partition is too large; split the window rather than silently truncate.')
            for hit in obj.get('hits',[]):
                text=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',hit.get('comment_text') or ''))).strip()
                timestamp=pd.to_datetime(hit['created_at'],utc=True)
                if len(text)<30 or not ELECTORAL.search(text) or not START<=timestamp<=CUTOFF:continue
                key=str(hit['objectID'])
                rows[key]={'document_id':'hn_'+key,'available_at':timestamp.isoformat(),'published_at':timestamp.isoformat(),
                           'source':'social','author_id':hit.get('author',''),'event_id':str(hit.get('story_id','')),
                           'text':text,'url':'https://news.ycombinator.com/item?id='+key}
            page+=1
            if page>=obj.get('nbPages',1):break
    pd.DataFrame(rows.values()).sort_values(['available_at','document_id']).to_csv(ROOT/'acquired/social.csv',index=False)
    print('Saved',len(rows),'electoral-context comments. Historical creation time does not prove unedited historical text.')

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',choices=['news','social'])
    args=parser.parse_args()
    # Any HTTP denial/rate limit stops this run; no bypass or automatic retry.
    {'news':fetch_news,'social':fetch_social}[args.source]()
