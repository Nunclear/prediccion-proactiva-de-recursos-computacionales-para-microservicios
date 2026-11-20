import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_results import exact_sign,crossed_bootstrap
import numpy as np
class StatisticsChecks(unittest.TestCase):
 def test_exact_sign_and_ties(self):
  self.assertEqual(exact_sign(np.ones(10)),2/1024)
  self.assertEqual(exact_sign(np.r_[np.ones(5),-np.ones(5)]),1.)
  self.assertEqual(exact_sign(np.zeros(10)),1.)
  self.assertEqual(exact_sign(np.r_[np.ones(10),0]),2/1024)
 def test_constant_paired_effect(self):
  a=np.full((10,20,83),.25)
  np.testing.assert_allclose(crossed_bootstrap(a,24,B=100),[.25,.25],atol=1e-14)
  a[0,0]=np.linspace(-2,2,83)
  lo,hi=crossed_bootstrap(a,24,B=200)
  lo2,hi2=crossed_bootstrap(-a,24,B=200)
  np.testing.assert_allclose([lo2,hi2],[-hi,-lo],atol=1e-14)
if __name__=='__main__':unittest.main()
