import unittest
import numpy as np
import pandas as pd
from research.evaluate_frozen import decision, evaluate, RULES


def sample(n=1500):
    idx = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
    close = 100*np.exp(np.cumsum(.001+np.sin(np.arange(n)/21)*.004))
    return pd.DataFrame({"open":close,"close":close,
                         "high":close*1.01,"low":close*.99}, index=idx)


class ResearchTests(unittest.TestCase):
    def test_signal_no_future_price_leakage(self):
        a=sample()
        b=a.copy()
        b.iloc[1400:, :] *= 1.7
        for rule in RULES:
            ta=decision(a, rule)
            tb=decision(b, rule)
            np.testing.assert_array_equal(ta[:1399],tb[:1399])

    def test_trade_only_after_signal_close(self):
        idx=pd.date_range("2022-01-01",periods=4,freq="h",tz="UTC")
        data=pd.DataFrame({"open":[100.,200.,100.,100.]},index=idx)
        # At close[1], buy. Cannot profit from open[0]->open[1].
        v=evaluate(data,np.array([0,1,0,0]),0,4,0)
        self.assertEqual(v["net_return"],0.)
        self.assertEqual(v["turnover"],2)

    def test_no_position_leverage(self):
        for rule in RULES:
            t=decision(sample(),rule)
            self.assertTrue(np.isin(t,[0,1]).all())

    def test_fee_applies_on_entry_and_exit(self):
        data=sample(20)
        t=np.zeros(20,dtype=np.int8)
        t[5:8]=1
        free=evaluate(data,t,0,20,0)
        costly=evaluate(data,t,0,20,10)
        self.assertLess(costly["net_return"],free["net_return"])
        self.assertEqual(costly["turnover"],2)


if __name__=="__main__":
    unittest.main()
