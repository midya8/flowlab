import copy
import json
import math
import unittest
from flowlab.core import Flow,train,artifact,load
from flowlab.demo import gaussian_fit,gaussian_log_prob,fixture


class FlowTests(unittest.TestCase):
    def test_every_parameter_gradient(self):
        for layers in [1,2,3]:
            n=Flow(layers,3,seed=4);x=[.3,-.8]
            _,g=n.loss_grad(x)
            for j in range(len(n.params)):
                old=n.params[j];n.params[j]=old+1e-6;a=n.loss_grad(x)[0]
                n.params[j]=old-1e-6;b=n.loss_grad(x)[0];n.params[j]=old
                self.assertAlmostEqual(g[j],(a-b)/2e-6,places=8)

    def test_full_jacobian_determinant_reference(self):
        for layers in [1,2,4]:
            n=Flow(layers,4);x=[.7,-1.1];jac=[]
            for k in range(2):
                a,b=list(x),list(x);a[k]+=1e-6;b[k]-=1e-6
                fa=n.forward(a)[0];fb=n.forward(b)[0]
                jac.append([(u-v)/2e-6 for u,v in zip(fa,fb)])
            determinant=jac[0][0]*jac[1][1]-jac[1][0]*jac[0][1]
            self.assertGreater(determinant,0.)
            self.assertAlmostEqual(math.log(determinant),n.forward(x)[1],places=8)

    def test_round_trip_both_directions_and_logdet(self):
        n=Flow()
        for x in fixture(50,1)+[[0.,0.],[20.,-20.]]:
            z,ld,_=n.forward(x);back,ild=n.inverse(z)
            for a,b in zip(x,back):self.assertAlmostEqual(a,b,places=11)
            self.assertAlmostEqual(ld+ild,0.,places=12)
            clean,ild=n.inverse(x);back,ld,_=n.forward(clean)
            for a,b in zip(x,back):self.assertAlmostEqual(a,b,places=11)
            self.assertAlmostEqual(ld+ild,0.,places=12)

    def test_identity_matches_standard_normal(self):
        n=Flow(2,3);n.params=[0.]*len(n.params)
        x=[.4,-.8]
        self.assertEqual(n.forward(x)[0],x)
        self.assertEqual(n.forward(x)[1],0.)
        self.assertAlmostEqual(n.log_prob(x),-math.log(2*math.pi)-sum(v*v for v in x)/2)

    def test_constant_affine_exact_density(self):
        n=Flow(1,1);n.params=[0.]*6
        # Raw scale output bias is at 3; translation bias at 5.
        n.params[3]=math.atanh(.4);n.params[5]=.7
        x=[.3,-.8];z=[x[0],x[1]*math.exp(.4)+.7]
        self.assertEqual(n.forward(x)[0],z)
        expected=-math.log(2*math.pi)-sum(v*v for v in z)/2+.4
        self.assertAlmostEqual(n.log_prob(x),expected)

    def test_density_numerically_integrates(self):
        n=Flow(2,3)
        # Deterministic midpoint quadrature, small near-identity nonlinear flow.
        width=12.;steps=120;delta=width/steps
        integral=sum(math.exp(n.log_prob([-6+(i+.5)*delta,-6+(j+.5)*delta]))*delta*delta
                     for i in range(steps) for j in range(steps))
        self.assertAlmostEqual(integral,1.,places=6)

    def test_scale_bounded_and_no_clipping(self):
        n=Flow(1,1,.5);n.params=[0.]*6;n.params[3]=1e6;n.params[5]=10.
        y,ld,_=n.forward([1.,100.])
        self.assertAlmostEqual(ld,.5)
        self.assertGreater(y[1],100.)
        self.assertAlmostEqual(n.inverse(y)[0][1],100.)

    def test_logdet_sign_essential(self):
        n=Flow(1,1);n.params=[0.]*6;n.params[3]=math.atanh(.5)
        z,ld,_=n.forward([0.,0.])
        self.assertEqual(z,[0.,0.]);self.assertAlmostEqual(ld,.5)
        self.assertAlmostEqual(n.log_prob([0.,0.]),-math.log(2*math.pi)+.5)
        self.assertNotEqual(n.log_prob([0.,0.]),-math.log(2*math.pi)-.5)

    def test_reproducible_fit_and_samples(self):
        x=fixture(12,1)
        a,ha=train(x,hidden=3,layers=2,updates=4,batch_size=3)
        b,hb=train(x,hidden=3,layers=2,updates=4,batch_size=3)
        self.assertEqual(a.params,b.params);self.assertEqual(ha,hb)
        self.assertEqual(a.sample(8,1),b.sample(8,1))
        self.assertNotEqual(a.sample(8,1),a.sample(8,2))

    def test_roundtrip_serialization(self):
        n=Flow();v=artifact(n);loaded=load(json.loads(json.dumps(v)))
        self.assertEqual(n.sample(8),loaded.sample(8))
        v['parameters'][0]=999.;self.assertNotEqual(n.params[0],999.)

    def test_strict_artifact(self):
        v=artifact(Flow());bad=[]
        for key in v:
            d=copy.deepcopy(v);del d[key];bad.append(d)
        d=copy.deepcopy(v);d['extra']=0;bad.append(d)
        d=copy.deepcopy(v);d['parameters'][0]=True;bad.append(d)
        d=copy.deepcopy(v);d['parameters'].pop();bad.append(d)
        d=copy.deepcopy(v);d['format']='wrong';bad.append(d)
        for d in bad:
            with self.assertRaises(ValueError):load(d)

    def test_invalid_model_settings(self):
        for kw in [{'layers':0},{'layers':9},{'hidden':True},{'scale_bound':0},{'scale_bound':float('nan')},{'seed':-1}]:
            with self.assertRaises(ValueError):Flow(**kw)

    def test_invalid_points(self):
        n=Flow()
        for x in [[1.],[True,0.],[float('inf'),0.],'ab']:
            with self.assertRaises(ValueError):n.log_prob(x)
            with self.assertRaises(ValueError):n.inverse(x)

    def test_training_boundaries(self):
        for kw in [{'updates':0},{'batch_size':129},{'learning_rate':.2}]:
            with self.assertRaises(ValueError):train([[0.,0.],[1.,1.]],**kw)
        for x in [[],[[0.,0.]],[[101.,0.],[0.,0.]]]:
            with self.assertRaises(ValueError):train(x,updates=1)

    def test_sample_boundaries(self):
        for count in [0,True,2049]:
            with self.assertRaises(ValueError):Flow().sample(count)

    def test_no_mutation(self):
        x=fixture(12,1);before=copy.deepcopy(x)
        n,_=train(x,updates=2);old=list(n.params)
        n.log_prob(x[0]);n.sample(4)
        self.assertEqual(x,before);self.assertEqual(n.params,old)

    def test_finite_extreme_log_density_error(self):
        with self.assertRaises(ValueError):Flow().log_prob([1e308,1e308])

    def test_loss_is_negative_log_density(self):
        n=Flow()
        for x in fixture(10,1):self.assertEqual(n.loss_grad(x)[0],-n.log_prob(x))

    def test_gaussian_fit_independent_reference(self):
        fit=gaussian_fit([[1.,0.],[-1.,0.],[0.,2.],[0.,-2.]])
        self.assertEqual(fit,([0.,0.],.5,0.,2.))
        self.assertAlmostEqual(gaussian_log_prob([1.,2.],fit),-math.log(2*math.pi)-2.)

    def test_alternating_coordinates(self):
        n=Flow(2,1);n.params=[0.]*12;n.params[5]=2.;n.params[11]=3.
        y,ld,_=n.forward([1.,1.])
        self.assertEqual(y,[4.,3.]);self.assertEqual(ld,0.)
        self.assertEqual(n.inverse(y)[0],[1.,1.])
