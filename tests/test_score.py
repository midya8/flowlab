import copy
import math
import unittest
from flowlab.core import Flow


class ScoreTests(unittest.TestCase):
    def test_both_coordinates_and_components_finite_differences(self):
        for layers in [1,2,3,8]:
            n=Flow(layers,3,.7,seed=17)
            # Stronger nonlinearity than the default near-identity initialization.
            n.params=[v*3 for v in n.params]
            for x in [[.3,-.8],[-1.2,.6],[0.,0.]]:
                r=n.input_score(x)
                for k in range(2):
                    a,b=list(x),list(x);a[k]+=1e-6;b[k]-=1e-6
                    za,lda,_=n.forward(a);zb,ldb,_=n.forward(b)
                    fd=(n.log_prob(a)-n.log_prob(b))/2e-6
                    self.assertAlmostEqual(r['score'][k],fd,places=7)
                    self.assertAlmostEqual(r['logdet_gradient'][k],(lda-ldb)/2e-6,places=7)
                    base=(-sum(z*z for z in za)/2+sum(z*z for z in zb)/2)/2e-6
                    self.assertAlmostEqual(r['base_pullback'][k],base,places=7)
                    self.assertEqual(r['score'][k],r['base_pullback'][k]+r['logdet_gradient'][k])
                self.assertEqual(r['log_density'],n.log_prob(x))

    def test_identity_score_is_negative_input(self):
        n=Flow(2,3);n.params=[0.]*len(n.params)
        r=n.input_score([.4,-.8])
        self.assertEqual(r['score'],[-.4,.8]);self.assertEqual(r['logdet_gradient'],[0.,0.])

    def test_constant_affine_score_closed_form(self):
        n=Flow(1,1);n.params=[0.]*6;n.params[3]=math.atanh(.4);n.params[5]=.7
        x=[.3,-.8];r=n.input_score(x)
        self.assertAlmostEqual(r['score'][0],-.3)
        self.assertAlmostEqual(r['score'][1],-(x[1]*math.exp(.4)+.7)*math.exp(.4))
        self.assertEqual(r['logdet_gradient'],[0.,0.])

    def test_variable_scale_closed_form_and_omission_counterexample(self):
        n=Flow(1,1);n.params=[.7,.1,.8,.2,-.4,.3]
        x=[.3,-.8];r=n.input_score(x);h=math.tanh(.7*x[0]+.1)
        s=math.tanh(.8*h+.2);ds=(1-s*s)*.8*(1-h*h)*.7
        t=-.4*h+.3;dt=-.4*(1-h*h)*.7;z=x[1]*math.exp(s)+t
        self.assertAlmostEqual(r['score'][0],-x[0]-z*(x[1]*math.exp(s)*ds+dt)+ds)
        self.assertAlmostEqual(r['score'][1],-z*math.exp(s))
        self.assertAlmostEqual(r['logdet_gradient'][0],ds)
        self.assertGreater(abs(r['score'][0]-r['base_pullback'][0]),.1)

    def test_gradient_does_not_mutate(self):
        n=Flow();x=[.2,.4];before=copy.deepcopy((n.params,x))
        n.input_score(x);self.assertEqual((n.params,x),before)

    def test_bad_inputs_and_numerical_failure(self):
        for x in [[True,0.],[float('nan'),0.],[1.], [1e308,1e308]]:
            with self.assertRaises(ValueError):Flow().input_score(x)
