"""Alternating 2-D affine coupling with exact likelihood and manual gradients."""
import math
import random


def integer(x,lo,hi,name):
    if type(x) is not int or not lo<=x<=hi:raise ValueError(f'{name} must be integer in [{lo},{hi}]')
    return x


def number(x,name):
    if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError(name+' must be finite')
    return float(x)


def point(x):
    if not isinstance(x,list) or len(x)!=2:raise ValueError('point needs two coordinates')
    return [number(v,'coordinate') for v in x]


class Flow:
    def __init__(self,layers=4,hidden=12,scale_bound=1.,seed=11):
        integer(layers,1,8,'layers');integer(hidden,1,32,'hidden');integer(seed,0,2**32-1,'seed')
        bound=number(scale_bound,'scale_bound')
        if not .01<=bound<=2:raise ValueError('scale_bound outside [.01,2]')
        self.layers,self.hidden,self.scale_bound=layers,hidden,bound
        r=random.Random(seed);self.params=[]
        for _ in range(layers):
            self.params += [r.gauss(0,.6) for _ in range(hidden)] + [r.gauss(0,.1) for _ in range(hidden)]
            self.params += [r.gauss(0,.03) for _ in range(hidden)] + [0.]
            self.params += [r.gauss(0,.03) for _ in range(hidden)] + [0.]

    def conditioner(self,a,offset):
        p,h=self.params,self.hidden
        act=[math.tanh(p[offset+j]*a+p[offset+h+j]) for j in range(h)]
        raw=sum(p[offset+2*h+j]*act[j] for j in range(h))+p[offset+3*h]
        unit=math.tanh(raw);scale=self.scale_bound*unit
        shift=sum(p[offset+3*h+1+j]*act[j] for j in range(h))+p[offset+4*h+1]
        return act,unit,scale,shift

    def forward(self,x):
        x=point(x);logdet=0.;cache=[];size=4*self.hidden+2
        try:
            for layer in range(self.layers):
                a,b=layer%2,1-layer%2;offset=layer*size
                act,unit,s,t=self.conditioner(x[a],offset)
                cache.append((a,b,x[a],x[b],act,unit,s))
                x[b]=x[b]*math.exp(s)+t;logdet+=s
                if not all(math.isfinite(v) for v in x):raise ValueError('nonfinite forward state')
        except OverflowError as e:raise ValueError('forward overflow') from e
        return x,logdet,cache

    def inverse(self,z):
        x=point(z);logdet=0.;size=4*self.hidden+2
        try:
            for layer in range(self.layers-1,-1,-1):
                a,b=layer%2,1-layer%2
                _,_,s,t=self.conditioner(x[a],layer*size)
                x[b]=(x[b]-t)*math.exp(-s);logdet-=s
                if not all(math.isfinite(v) for v in x):raise ValueError('nonfinite inverse state')
        except OverflowError as e:raise ValueError('inverse overflow') from e
        return x,logdet

    def log_prob(self,x):
        z,ld,_=self.forward(x)
        value=-math.log(2*math.pi)-sum(v*v for v in z)/2+ld
        if not math.isfinite(value):raise ValueError('nonfinite log density')
        return value

    def loss_grad(self,x):
        z,ld,cache=self.forward(x);h=self.hidden;p=self.params;size=4*h+2
        loss=math.log(2*math.pi)+sum(v*v for v in z)/2-ld
        dx=list(z);g=[0.]*len(p)
        for layer in range(self.layers-1,-1,-1):
            a,b,xa,xb,act,unit,s=cache[layer];offset=layer*size
            ds=dx[b]*xb*math.exp(s)-1.
            dt=dx[b];dr=ds*self.scale_bound*(1-unit*unit)
            da=0.
            for j in range(h):
                g[offset+2*h+j]=dr*act[j]
                g[offset+3*h+1+j]=dt*act[j]
                dh=(dr*p[offset+2*h+j]+dt*p[offset+3*h+1+j])*(1-act[j]*act[j])
                g[offset+j]=dh*xa;g[offset+h+j]=dh;da+=dh*p[offset+j]
            g[offset+3*h]=dr;g[offset+4*h+1]=dt
            dx[a]+=da;dx[b]*=math.exp(s)
        if not math.isfinite(loss) or not all(math.isfinite(v) for v in g):raise ValueError('nonfinite loss/gradient')
        return loss,g

    def sample(self,count=256,seed=19):
        integer(count,1,2048,'count');integer(seed,0,2**32-1,'seed')
        r=random.Random(seed)
        return [self.inverse([r.gauss(0,1),r.gauss(0,1)])[0] for _ in range(count)]


def train(points,layers=4,hidden=12,updates=2000,batch_size=32,learning_rate=.003,seed=11):
    if not isinstance(points,list) or not 2<=len(points)<=8192:raise ValueError('need 2..8192 training points')
    points=[point(x) for x in points]
    if any(abs(v)>100 for row in points for v in row):raise ValueError('training coordinates outside [-100,100]')
    integer(updates,1,10000,'updates');integer(batch_size,1,128,'batch_size')
    lr=number(learning_rate,'learning_rate')
    if not 1e-6<=lr<=.1:raise ValueError('learning rate outside [1e-6,.1]')
    net=Flow(layers,hidden,seed=seed);r=random.Random(seed+1)
    m,v=[0.]*len(net.params),[0.]*len(net.params);history=[]
    for step in range(1,updates+1):
        loss=0.;g=[0.]*len(net.params)
        for _ in range(batch_size):
            value,grad=net.loss_grad(r.choice(points));loss+=value/batch_size
            for j,z in enumerate(grad):g[j]+=z/batch_size
        for j,z in enumerate(g):
            m[j]=.9*m[j]+.1*z;v[j]=.999*v[j]+.001*z*z
            net.params[j]-=lr*(m[j]/(1-.9**step))/(math.sqrt(v[j]/(1-.999**step))+1e-8)
        if not all(math.isfinite(z) for z in net.params):raise ValueError('training diverged')
        if step==1 or step%250==0 or step==updates:history.append({'update':step,'batch_nll_before_update':loss})
    return net,history


def artifact(net):
    return {'format':'flowlab_v1','layers':net.layers,'hidden':net.hidden,
            'scale_bound':net.scale_bound,'parameters':list(net.params)}


def load(v):
    if not isinstance(v,dict) or set(v)!={'format','layers','hidden','scale_bound','parameters'} or v['format']!='flowlab_v1':raise ValueError('invalid model fields/version')
    n=Flow(v['layers'],v['hidden'],v['scale_bound'])
    if not isinstance(v['parameters'],list) or len(v['parameters'])!=len(n.params):raise ValueError('invalid parameter count')
    n.params=[number(x,'parameter') for x in v['parameters']]
    return n
