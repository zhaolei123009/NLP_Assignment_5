"""Time-bounded TF-IDF/NMF discovery; topic weights are not probabilities."""
import hashlib,re
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate(documents:pd.DataFrame,near_threshold=0.93):
    f=documents.copy()
    if f.document_id.duplicated().any():raise ValueError('Duplicate document IDs; resolve source identity first')
    f['available_at']=pd.to_datetime(f.available_at,utc=True)
    f=f.sort_values(['available_at','document_id'])
    f['original_text']=f.text
    # Syndicated radio/news stories append station branding after a pipe.
    # Preserve raw titles, but do not treat station suffixes as independent events.
    f['text']=f.text.fillna('').map(lambda x:str(x).split('|',1)[0].strip())
    f['normalized_text']=f.text.fillna('').map(lambda x:re.sub(r'\s+',' ',re.sub(r'[^a-z0-9 ]',' ',str(x).lower())).strip())
    f=f.loc[f.normalized_text.str.len()>=20].drop_duplicates('normalized_text').copy()
    if len(f)>1:
        v=TfidfVectorizer(ngram_range=(1,2),min_df=1).fit_transform(f.normalized_text)
        # Local event similarity only; same language months apart can be independent observations.
        keep=[]; times=f.available_at.tolist()
        for i in range(len(f)):
            prior=[j for j in keep if times[i]-times[j]<=pd.Timedelta(days=3)]
            if not prior or cosine_similarity(v[i],v[prior]).max()<near_threshold:keep.append(i)
        f=f.iloc[keep].copy()
    f['event_id']=f.get('event_id',pd.Series('',index=f.index)).fillna('')
    missing=f.event_id.eq('');f.loc[missing,'event_id']=f.loc[missing,'normalized_text'].map(lambda t:hashlib.sha256(t.encode()).hexdigest()[:16])
    return f.drop(columns='normalized_text').reset_index(drop=True)

class TopicDiscovery:
    def fit(self,documents,train_until,n_topics=6):
        cutoff=pd.Timestamp(train_until)
        train=documents.loc[pd.to_datetime(documents.available_at,utc=True)<=cutoff].copy()
        if len(train)<40:raise ValueError('Need at least 40 training documents before cutoff')
        stop=sorted(set(ENGLISH_STOP_WORDS)|{'news','ap','edt','radio','newsradio','newstalk','click','read'})
        self.vectorizer=TfidfVectorizer(stop_words=stop,token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b',ngram_range=(1,2),min_df=2,max_df=0.9,max_features=15000)
        x=self.vectorizer.fit_transform(train.text)
        self.model=NMF(n_components=min(n_topics,len(train)-1,x.shape[1]),init='nndsvda',random_state=2026,max_iter=500)
        self.model.fit(x);self.train_until=cutoff.isoformat();self.train_n=len(train)
        return self

    def transform(self,documents):
        w=self.model.transform(self.vectorizer.transform(documents.text));total=w.sum(axis=1)
        probs=np.divide(w,total[:,None],out=np.zeros_like(w),where=total[:,None]>0)
        f=pd.DataFrame(probs,columns=[f'nmf_topic_{i}' for i in range(w.shape[1])]);f.insert(0,'document_id',documents.document_id.to_numpy())
        return f

    def description(self,n_terms=10):
        terms=self.vectorizer.get_feature_names_out()
        return {'model':'TF-IDF + NMF; weights are not calibrated topic probabilities',
                'train_until':self.train_until,'train_n':self.train_n,
                'topics':[{'id':i,'terms':terms[np.argsort(row)[-n_terms:][::-1]].tolist()} for i,row in enumerate(self.model.components_)]}

def discover_news(documents):
    cleaned=deduplicate(documents.fillna(''))
    times=pd.to_datetime(cleaned.available_at,utc=True).sort_values()
    cutoff=times.iloc[int(len(times)*.65)-1]
    model=TopicDiscovery().fit(cleaned,cutoff,n_topics=6)
    assignments=model.transform(cleaned)
    info=model.description()
    info.update(raw_documents=len(documents),deduplicated_documents=len(cleaned),
                heldout_documents=int((pd.to_datetime(cleaned.available_at,utc=True)>cutoff).sum()),
                descriptive_topic_counts=assignments.iloc[:,1:].idxmax(axis=1).value_counts().to_dict())
    return cleaned,info
