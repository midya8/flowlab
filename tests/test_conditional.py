import copy
import math
import unittest
from flowlab.core import Flow
from flowlab.conditional import conditional_slice


class Constant:
    def __init__(self,value=-10000.):self.calls=0;self.value=value
    def log_prob(self,x):self.calls+=1;return self.value


class ConditionalTests(unittest.TestCase):
    def test_uniform_exact_moments_cdf_quantiles_and_log_stability(self):
        for log_value in [-10000.,10000.]:
            n=Constant(log_value);r=conditional_slice(n,0,.2,-2.,4.,7,[0.,.25,.5,.75,1.])
            self.assertEqual(n.calls,7);self.assertEqual(r['model_evaluations'],7)
            self.assertAlmostEqual(r['mean'],1.);self.assertAlmostEqual(r['variance'],3.)
            self.assertAlmostEqual(r['log_slice_integral_approx'],log_value+math.log(6))
            for row,value in zip(r['quantiles'],[-2.,-.5,1.,2.5,4.]):self.assertAlmostEqual(row['value'],value)
            for i,v in enumerate(r['cdf']):self.assertAlmostEqual(v,i/6)

    def test_triangular_linear_density_exact(self):
        class Linear:
            def log_prob(self,x):return math.log(x[1]+1)
        # Density proportional to x+1 on [0,2]. Trapezoid is exact for any grid.
        for size in [2,3,33]:
            r=conditional_slice(Linear(),0,0.,0.,2.,size,[.5])
            self.assertAlmostEqual(r['mean'],7/6)
            self.assertAlmostEqual(r['variance'],11/36)
            self.assertAlmostEqual(r['quantiles'][0]['value'],math.sqrt(5)-1)

    def test_normal_slice_and_refinement(self):
        n=Flow(2,1);n.params=[0.]*12
        coarse=conditional_slice(n,0,.7,-6.,6.,65,[.025,.5,.975])
        fine=conditional_slice(n,0,.7,-6.,6.,513,[.025,.5,.975])
        self.assertAlmostEqual(fine['mean'],0.,places=12)
        self.assertAlmostEqual(fine['variance'],1.,places=3)
        self.assertLess(abs(fine['quantiles'][2]['value']-1.95996398454),abs(coarse['quantiles'][2]['value']-1.95996398454))
        self.assertAlmostEqual(fine['quantiles'][1]['value'],0.,places=12)
        self.assertAlmostEqual(fine['log_slice_integral_approx'],-math.log(2*math.pi)/2-.7*.7/2,places=7)

    def test_truncation_is_visible_not_standard_normal(self):
        n=Flow(1,1);n.params=[0.]*6
        r=conditional_slice(n,0,0.,0.,1.,513)
        expected=(math.exp(0)-math.exp(-.5))/math.sqrt(2*math.pi)/(.5*math.erf(1/math.sqrt(2)))
        self.assertAlmostEqual(r['mean'],expected,places=6)
        self.assertGreater(r['mean'],.4);self.assertLess(r['variance'],.1)
        self.assertEqual(r['interval'],[0.,1.]);self.assertIn('tails unknown',r['semantics'])

    def test_alternate_axis_and_constant_affine_normal(self):
        n=Flow(1,1);n.params=[0.]*6;n.params[3]=math.atanh(.4);n.params[5]=.7
        r=conditional_slice(n,0,.2,-6.,6.,513)
        self.assertAlmostEqual(r['mean'],-.7*math.exp(-.4),places=7)
        self.assertAlmostEqual(r['variance'],math.exp(-.8),places=3)
        r=conditional_slice(n,1,.2,-6.,6.,513)
        self.assertAlmostEqual(r['mean'],0.,places=12)
        self.assertAlmostEqual(r['variance'],1.,places=3)

    def test_preflight_limits_make_zero_calls(self):
        for kw in [{'fixed_axis':True},{'fixed_axis':2},{'fixed_value':101},{'low':4.,'high':4.},
                   {'low':-101.},{'high':float('inf')},{'grid_size':1},{'grid_size':514},
                   {'grid_size':True},{'quantiles':[]},{'quantiles':[True]},{'quantiles':[1.1]},
                   {'quantiles':[.5]*34}]:
            n=Constant();args={'fixed_axis':0,'fixed_value':0.,**kw}
            with self.assertRaises(ValueError):conditional_slice(n,**args)
            self.assertEqual(n.calls,0)

    def test_nonfinite_density_fails(self):
        for value in [float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):conditional_slice(Constant(value),0,0.)

    def test_no_mutation_and_monotone_cdf(self):
        n=Flow();before=copy.deepcopy(n.params);q=[.95,.05,.5,.5]
        r=conditional_slice(n,1,.7,quantiles=q)
        self.assertEqual(n.params,before);self.assertEqual(q,[.95,.05,.5,.5])
        self.assertEqual([v['probability'] for v in r['quantiles']],q)
        self.assertEqual(r['quantiles'][2],r['quantiles'][3])
        self.assertTrue(all(a<=b for a,b in zip(r['cdf'],r['cdf'][1:])))
        area=sum((b-a)*(u+v)/2 for a,b,u,v in zip(r['grid'],r['grid'][1:],r['conditional_density'],r['conditional_density'][1:]))
        self.assertAlmostEqual(area,1.)

    def test_conditional_cli(self):
        import json,subprocess,sys,tempfile,os
        from flowlab.core import artifact
        with tempfile.TemporaryDirectory() as d:
            model=os.path.join(d,'model.json')
            with open(model,'w') as f:json.dump(artifact(Flow()),f)
            for size,code in [('33',0),('514',2)]:
                p=subprocess.run([sys.executable,'-m','flowlab','conditional',model,'--axis','1','--value','.7','--grid-size',size],capture_output=True,text=True)
                self.assertEqual(p.returncode,code,p.stderr)
                if code==0:
                    r=json.loads(p.stdout);self.assertEqual(r['model_evaluations'],33);self.assertEqual(r['free_axis'],0)
                else:self.assertNotIn('Traceback',p.stderr)
