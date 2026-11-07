"""Finite-interval conditional slice, with an explicit piecewise-linear density."""
import math
from .core import integer,number


def conditional_slice(net,fixed_axis,fixed_value,low=-4.,high=5.,grid_size=257,quantiles=None):
    """Normalize a sampled joint-density slice on [low,high], not the real line.

    The trapezoid rule integrates a piecewise-linear density exactly; all moments
    and quantiles describe that interpolation after finite-interval renormalizing.
    No tail probability, true marginal likelihood or accuracy bound is inferred.
    """
    axis=integer(fixed_axis,0,1,'fixed_axis');fixed=number(fixed_value,'fixed_value')
    lo,hi=number(low,'low'),number(high,'high');n=integer(grid_size,2,513,'grid_size')
    if not -100<=lo<hi<=100 or hi-lo<1e-6:raise ValueError('need -100<=low<high<=100 and width>=1e-6')
    if not -100<=fixed<=100:raise ValueError('fixed value outside [-100,100]')
    q=[.05,.5,.95] if quantiles is None else quantiles
    if not isinstance(q,list) or not 1<=len(q)<=33:raise ValueError('need 1..33 quantiles')
    q=[number(v,'quantile') for v in q]
    if any(not 0<=v<=1 for v in q):raise ValueError('quantile outside [0,1]')
    # Validation finishes before the first model evaluation. No hidden search.
    xs=[lo+(hi-lo)*i/(n-1) for i in range(n)];xs[-1]=hi
    logs=[]
    for x in xs:
        point=[0.,0.];point[axis]=fixed;point[1-axis]=x
        logs.append(number(net.log_prob(point),'slice log density'))
    return {'fixed_axis':axis,'fixed_value':fixed,'free_axis':1-axis,
            'model_evaluations':n,**_summarize(xs,logs,q)}


def _summarize(xs,logs,q):
    lo,hi,n=xs[0],xs[-1],len(xs)
    shift=max(logs);ys=[math.exp(v-shift) for v in logs]
    areas=[(xs[i+1]-xs[i])*(ys[i]+ys[i+1])/2 for i in range(n-1)]
    z=math.fsum(areas)
    if not z>0 or not math.isfinite(z):raise ValueError('invalid slice normalizer')
    density=[v/z for v in ys];cdf=[0.];cumulative=0.
    first=second=0.
    for i,area in enumerate(areas):
        a,b=ys[i],ys[i+1];x=xs[i];h=xs[i+1]-x
        first+=h*(x*(a+b)/2+h*(a/6+b/3))
        second+=h*(x*x*(a+b)/2+2*x*h*(a/6+b/3)+h*h*(a/12+b/4))
        cumulative+=area;cdf.append(min(1.,cumulative/z))
    cdf[-1]=1.;mean=first/z;variance=max(0.,second/z-mean*mean)
    values=[]
    for probability in q:
        if probability==0:value=lo
        elif probability==1:value=hi
        else:
            i=0
            while i<n-2 and cdf[i+1]<probability:i+=1
            target=(probability-cdf[i])*z;h=xs[i+1]-xs[i];a,b=ys[i],ys[i+1]
            left,right=0.,1.
            for _ in range(60):
                mid=(left+right)/2
                if h*(a*mid+(b-a)*mid*mid/2)<target:left=mid
                else:right=mid
            value=xs[i]+h*(left+right)/2
        values.append({'probability':probability,'value':value})
    normalizer=shift+math.log(z)
    if not all(math.isfinite(v) for v in [normalizer,mean,variance]+density):raise ValueError('nonfinite conditional result')
    return {'interval':[lo,hi],'grid_size':n,'grid':xs,'joint_log_density':logs,
            'conditional_density':density,'cdf':cdf,'quantiles':values,'mean':mean,'variance':variance,
            'log_slice_integral_approx':normalizer,
            'semantics':'piecewise-linear sampled joint density, renormalized on finite interval; tails unknown'}


def window_audit(net,fixed_axis,fixed_value,low=-4.,high=4.,grid_size=513,inner_start=192,inner_end=320):
    """Compare a grid-aligned inner interval against a finite outer interval.

    Only the outer grid is evaluated. The inner density is re-normalized from
    the identical sampled heights. This diagnoses window sensitivity of the
    interpolant, not missing peaks, whole-line mass or approximation accuracy.
    """
    n=integer(grid_size,2,513,'grid_size')
    start=integer(inner_start,0,n-2,'inner_start');end=integer(inner_end,start+1,n-1,'inner_end')
    outer=conditional_slice(net,fixed_axis,fixed_value,low,high,n)
    inner=_summarize(outer['grid'][start:end+1],outer['joint_log_density'][start:end+1],[.05,.5,.95])
    # Compute ratio directly in the outer log scale, avoiding subtraction of
    # large shared log offsets in the two reported log integrals.
    logs=outer['joint_log_density'];shift=max(logs);heights=[math.exp(v-shift) for v in logs]
    areas=[(outer['grid'][i+1]-outer['grid'][i])*(heights[i]+heights[i+1])/2 for i in range(n-1)]
    ratio=math.fsum(areas[start:end])/math.fsum(areas)
    return {'fixed_axis':outer['fixed_axis'],'fixed_value':outer['fixed_value'],
            'free_axis':outer['free_axis'],'model_evaluations':n,'inner_grid_indices':[start,end],
            'outer':outer,'inner':inner,'interpolated_inner_mass_within_outer':ratio,
            'interpolated_removed_fraction_within_outer':1-ratio,
            'median_shift_inner_minus_outer':inner['quantiles'][1]['value']-outer['quantiles'][1]['value'],
            'semantics':'mass fractions of the sampled piecewise-linear density within the finite outer interval; whole-line tails and missed peaks unknown'}
