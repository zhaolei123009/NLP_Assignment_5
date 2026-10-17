"""Critical integrity checks with synthetic fixtures, not measured model accuracy."""
import unittest
import numpy as np
import pandas as pd
from sentiment import check_cutoff,aggregate_daily,basket_panel,relation,score_documents
from equities import asof_values
from fetch_inputs import ELECTORAL

class CoreIntegrity(unittest.TestCase):
    def test_future_documents_rejected(self):
        with self.assertRaises(ValueError):
            check_cutoff(pd.DataFrame({'document_id':['x'],'available_at':['2026-10-08T00:00:00Z']}))

    def test_exam_midterms_rejected(self):
        self.assertIsNone(ELECTORAL.search('My CS midterm and final exams determine the course grade.'))
        self.assertIsNotNone(ELECTORAL.search('Trump and Democrats campaign for midterm elections.'))

    def test_quote_requires_full_bucket_embargo(self):
        market=pd.DataFrame({'timestamp':['2026-07-01T12:00:00Z'],'probability':[.6],'resolution_seconds':[10800]})
        times=pd.to_datetime(['2026-07-01T14:59:00Z','2026-07-01T15:00:00Z'],utc=True)
        r=asof_values(market,times)
        self.assertTrue(pd.isna(r.probability.iloc[0]));self.assertEqual(r.probability.iloc[1],.6)

    def test_equal_author_weight_and_partial_day(self):
        rows=[{'source':'social','topic':'pooled','available_at':'2026-07-01T12:00:00Z','author_id':'a','event_id':'s','compound':1.0}]*20
        rows += [{'source':'social','topic':'pooled','available_at':'2026-07-01T13:00:00Z','author_id':'b','event_id':'t','compound':-1.0},
                 {'source':'social','topic':'pooled','available_at':'2026-10-07T12:00:00Z','author_id':'a','event_id':'s','compound':1.0}]
        r=aggregate_daily(pd.DataFrame(rows));r=r[r.topic=='pooled'].set_index('day')
        self.assertEqual(r.loc[pd.Timestamp('2026-07-01',tz='UTC'),'sentiment'],0)
        self.assertTrue(pd.isna(r.loc[pd.Timestamp('2026-10-07',tz='UTC'),'sentiment']))

    def test_missing_day_is_not_replaced_by_older_signal(self):
        days=pd.to_datetime(['2026-07-06','2026-07-07','2026-07-08'])
        basket=pd.DataFrame({'equal_weight_basket_ar':[.01,.02,.03],'pro_democratic_spread_ar':[.02,.03,.04],'spy_return':[.01,.01,.01]},index=days)
        series=pd.DataFrame({'available_at':pd.to_datetime(['2026-07-06T00:00Z','2026-07-07T00:00Z','2026-07-08T00:00Z']),
                             'sentiment':[.5,np.nan,.9],'comment_count':[3,0,3],'authors':[3,0,3],'stories':[2,0,2],'low_coverage':[False,True,False]})
        result=basket_panel(series,basket)
        self.assertEqual(result.loc['2026-07-07','sentiment_signal'],.5)
        self.assertTrue(pd.isna(result.loc['2026-07-08','sentiment_signal']))

    def test_small_samples_have_no_hac_pvalue(self):
        r=relation(pd.DataFrame({'y':[1,2,4,3,5],'x':[1,3,2,4,5]}),'y',['x'])
        self.assertIsNotNone(r['pearson']);self.assertIsNone(r['pvalue'])

    def test_policy_does_not_match_crime(self):
        r=score_documents(pd.DataFrame([{'document_id':'x','available_at':'2026-07-01T12:00Z','text':'Federal policy is a good idea.',
                                        'source':'social','author_id':'a','event_id':'s','url':'https://example.org'}]))
        self.assertNotIn('crime',r.topic.tolist())

if __name__=='__main__':unittest.main()
