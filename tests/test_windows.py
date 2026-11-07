import math
import unittest
from flowlab.core import Flow
from flowlab.conditional import conditional_slice,window_audit


class Counter:
    def __init__(self,value=0.):self.calls=0;self.value=value
    def log_prob(self,x):self.calls+=1;return self.value


class WindowTests(unittest.TestCase):
    def test_uniform_ratio_quantile_shift_and_single_grid_budget(self):
        for offset in [-10000.,0.,10000.]:
            n=Counter(offset);r=window_audit(n,0,0.,-2.,4.,7,1,4)
            self.assertEqual(n.calls,7);self.assertEqual(r['model_evaluations'],7)
            self.assertEqual(r['inner']['interval'],[-1.,2.]);self.assertEqual(r['inner']['grid_size'],4)
            self.assertEqual(r['inner']['grid'],r['outer']['grid'][1:5])
            self.assertEqual(r['inner']['joint_log_density'],r['outer']['joint_log_density'][1:5])
            self.assertEqual(r['interpolated_inner_mass_within_outer'],.5)
            self.assertEqual(r['interpolated_removed_fraction_within_outer'],.5)
            self.assertAlmostEqual(r['median_shift_inner_minus_outer'],-.5)
            self.assertAlmostEqual(r['inner']['mean'],.5)
            self.assertAlmostEqual(r['inner']['variance'],.75)

    def test_full_window_is_identical(self):
        n=Flow();r=window_audit(n,1,.7,-3.,3.,65,0,64)
        self.assertEqual(r['interpolated_inner_mass_within_outer'],1.)
        self.assertEqual(r['interpolated_removed_fraction_within_outer'],0.)
        for key in r['inner']:self.assertEqual(r['inner'][key],r['outer'][key])

    def test_truncated_normal_ratio_against_erf(self):
        n=Flow(1,1);n.params=[0.]*6
        r=window_audit(n,0,.3,-4.,4.,513,192,320)
        true_ratio=math.erf(1/math.sqrt(2))/math.erf(4/math.sqrt(2))
        self.assertAlmostEqual(r['interpolated_inner_mass_within_outer'],true_ratio,places=4)
        self.assertLess(r['inner']['variance'],r['outer']['variance'])

    def test_nested_interpolant_mass_monotonicity(self):
        n=Flow();a=window_audit(n,0,.7,-4.,4.,129,32,96);b=window_audit(n,0,.7,-4.,4.,129,48,80)
        self.assertGreater(a['interpolated_inner_mass_within_outer'],b['interpolated_inner_mass_within_outer'])
        self.assertEqual(a['outer'],b['outer'])

    def test_invalid_indices_preflight_zero_calls(self):
        for args in [(True,20),(-1,20),(10,10),(10,9),(0,33),(32,33)]:
            n=Counter()
            with self.assertRaises(ValueError):window_audit(n,0,0.,-1.,1.,33,*args)
            self.assertEqual(n.calls,0)
        n=Counter()
        with self.assertRaises(ValueError):window_audit(n,0,0.,grid_size=514)
        self.assertEqual(n.calls,0)

    def test_grid_agreement_can_miss_narrow_peak(self):
        class HiddenPeak:
            # Broad almost-uniform component plus a normalized narrow bump.
            # Both 17/33 grids omit x=.015625. True interval mass is about .5001,
            # not the trapezoid estimate .0001. No neural-model claim here.
            def log_prob(self,x):
                sd=.0001;peak=.5*math.exp(-.5*((x[1]-.015625)/sd)**2)/(sd*math.sqrt(2*math.pi))
                return math.log(.0001+peak)
        results=[conditional_slice(HiddenPeak(),0,0.,0.,1.,n) for n in [17,33]]
        self.assertAlmostEqual(results[0]['mean'],results[1]['mean'])
        self.assertAlmostEqual(results[0]['log_slice_integral_approx'],results[1]['log_slice_integral_approx'])
        self.assertAlmostEqual(math.exp(results[1]['log_slice_integral_approx']),.0001)
        self.assertGreater(.5001/math.exp(results[1]['log_slice_integral_approx']),5000)
        expected_mean=(.0001*.5+.5*.015625)/.5001
        self.assertGreater(abs(results[1]['mean']-expected_mean),.45)

    def test_cli_window_limits_and_output(self):
        import json,subprocess,sys,tempfile,os
        from flowlab.core import artifact
        with tempfile.TemporaryDirectory() as d:
            pth=os.path.join(d,'flow.json')
            with open(pth,'w') as f:json.dump(artifact(Flow()),f)
            for end,code in [('320',0),('900',2)]:
                p=subprocess.run([sys.executable,'-m','flowlab','windows',pth,'--axis','0','--value','0','--inner-end',end],capture_output=True,text=True)
                self.assertEqual(p.returncode,code,p.stderr)
                if code==0:self.assertEqual(json.loads(p.stdout)['model_evaluations'],513)
                else:self.assertNotIn('Traceback',p.stderr)
