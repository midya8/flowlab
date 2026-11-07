import math
import random
from .core import Flow,train


def fixture(n,seed):
    r=random.Random(seed);out=[]
    for _ in range(n):
        u,v=r.gauss(0,1),r.gauss(0,1)
        out.append([u,.5*v+.6*(u*u-1)])
    return out


def gaussian_fit(points):
    n=len(points);mu=[sum(x[j] for x in points)/n for j in range(2)]
    a=sum((x[0]-mu[0])**2 for x in points)/n
    b=sum((x[0]-mu[0])*(x[1]-mu[1]) for x in points)/n
    c=sum((x[1]-mu[1])**2 for x in points)/n
    return mu,a,b,c


def gaussian_log_prob(x,fit):
    mu,a,b,c=fit;det=a*c-b*b;u,v=x[0]-mu[0],x[1]-mu[1]
    return -math.log(2*math.pi)-.5*math.log(det)-.5*(c*u*u-2*b*u*v+a*v*v)/det


def sample_summary(points):
    residual=[x[1]-.6*(x[0]*x[0]-1) for x in points]
    return {'mean_x':sum(x[0] for x in points)/len(points),
            'mean_y':sum(x[1] for x in points)/len(points),
            'mean_curve_residual':sum(residual)/len(points),
            'curve_residual_second_moment':sum(v*v for v in residual)/len(points),
            'fraction_abs_curve_residual_le_1':sum(abs(v)<=1 for v in residual)/len(points)}


def demo():
    points,test=fixture(1024,53),fixture(512,59)
    net,history=train(points)
    initial=Flow();fit=gaussian_fit(points)
    models={'trained_flow':net.log_prob,'untrained_flow':initial.log_prob,
            'fitted_gaussian':lambda x:gaussian_log_prob(x,fit),
            'generating_reference':lambda x:-math.log(2*math.pi)-.5*(x[0]**2+((x[1]-.6*(x[0]**2-1))/.5)**2)-math.log(.5)}
    nll={name:{'train':-sum(fn(x) for x in points)/len(points),'test':-sum(fn(x) for x in test)/len(test)} for name,fn in models.items()}
    r=random.Random(19);mu,a,b,c=fit
    gp=[]
    for _ in range(256):
        z,w=r.gauss(0,1),r.gauss(0,1)
        gp.append([mu[0]+math.sqrt(a)*z,mu[1]+b/math.sqrt(a)*z+math.sqrt(c-b*b/a)*w])
    samples={'trained_flow':net.sample(),'untrained_flow':initial.sample(),'fitted_gaussian':gp,'reference':fixture(256,61)}
    errors=[]
    for x in test:
        z,ld,_=net.forward(x);back,ild=net.inverse(z)
        errors.append(max(abs(back[j]-x[j]) for j in range(2)))
        errors.append(abs(ld+ild))
    result={'fixture':{'train_seed':53,'test_seed':59,'reference_sample_seed':61,'sample_seed':19,
                       'network_seed':11,'training_draw_seed':12,'training_count':1024,'test_count':512,
                       'layers':4,'hidden':12,'parameters':len(net.params),'updates':2000,'batch_size':32,'learning_rate':.003},
            'nll_nats_per_2d_point':nll,'history':history,'gaussian':{'mean':mu,'covariance':[[a,b],[b,c]]},
            'max_round_trip_or_logdet_residual':max(errors),
            'sample_summary':{name:sample_summary(x) for name,x in samples.items()},'samples':samples}
    result['input_score_audit']=score_audit(net)
    result['conditional_audit']=conditional_audit(net,fit)
    result['window_audit']=window_demo(net)
    assert nll['trained_flow']['test'] < nll['fitted_gaussian']['test']
    assert max(errors)<1e-10
    return result


def score_audit(net):
    # Fixed grid, selected before computing score errors. No optimization here.
    rows=[];errors=[];omission=[]
    for i in range(9):
        for j in range(9):
            x=[-2.+i*.5,-1.+j*.5];r=net.input_score(x);fd=[]
            for k in range(2):
                plus,minus=list(x),list(x);plus[k]+=1e-5;minus[k]-=1e-5
                fd.append((net.log_prob(plus)-net.log_prob(minus))/2e-5)
            rows.append({'point':x,**r,'central_difference':fd})
            errors.extend(abs(a-b) for a,b in zip(r['score'],fd))
            omission.extend(abs(a-b) for a,b in zip(r['base_pullback'],fd))
    assert max(errors)<1e-6
    assert max(omission)>.1
    return {'grid':'9x9, x=-2..2, y=-1..3, spacing .5; epsilon 1e-5',
            'max_abs_coordinate_error':max(errors),
            'max_abs_error_omitting_logdet_gradient':max(omission),'rows':rows}


def conditional_audit(net,fit):
    from .conditional import conditional_slice
    rows=[];mu,a,b,c=fit
    for fixed in [-1.5,0.,1.5]:
        slices=[conditional_slice(net,0,fixed,-4.,5.,n) for n in [65,257,513]]
        rows.append({'fixed_x':fixed,'slices':slices,
                     'generating_mean':.6*(fixed*fixed-1),'generating_sd':.5,
                     'gaussian_conditional_mean':mu[1]+b/a*(fixed-mu[0]),
                     'gaussian_conditional_sd':math.sqrt(c-b*b/a),
                     'median_change_257_to_513':abs(slices[1]['quantiles'][1]['value']-slices[2]['quantiles'][1]['value'])})
    assert max(row['median_change_257_to_513'] for row in rows)<.002
    return {'interval':[-4.,5.],'grid_sizes':[65,257,513],'total_model_evaluations':3*(65+257+513),
            'reference_note':'known generator and fitted Gaussian formulas describe full real-line conditionals; grid flow is truncated',
            'rows':rows}


def window_demo(net):
    from .conditional import window_audit
    rows=[window_audit(net,0,x) for x in [-1.5,0.,1.5]]
    assert all(.5<r['interpolated_inner_mass_within_outer']<.9 for r in rows)
    # An analytic non-neural adversarial integrand, not another learned model.
    class HiddenPeak:
        def log_prob(self,x):
            sd=.0001
            peak=.5*math.exp(-.5*((x[1]-.015625)/sd)**2)/(sd*math.sqrt(2*math.pi))
            return math.log(.0001+peak)
    from .conditional import conditional_slice
    missed=[conditional_slice(HiddenPeak(),0,0.,0.,1.,n) for n in [17,33]]
    reference_mass=.0001+.5*.5*(math.erf((1-.015625)/(.0001*math.sqrt(2)))-math.erf(-.015625/(.0001*math.sqrt(2))))
    assert all(abs(math.exp(s['log_slice_integral_approx'])-.0001)<1e-12 for s in missed)
    assert reference_mass>.5
    return {'rows':rows,'total_model_evaluations':sum(r['model_evaluations'] for r in rows),
            'missed_peak':{'non_neural_density':'.0001 + .5*NormalPDF(y; mean=.015625, sd=.0001), y in[0,1]',
                           'slices':missed,'reference_interval_integral':reference_mass,
                           'counterexample_evaluations':50},
            'limits':'finite outer interval only; mass ratios describe interpolation, not true whole-line probabilities'}
