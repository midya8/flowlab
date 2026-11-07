import argparse
import json
from .core import train,load,artifact,point
from .demo import demo


def read(p):
    with open(p,encoding='utf-8') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description='Explicit offline 2-D affine coupling flow')
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('demo')
    t=sub.add_parser('train');t.add_argument('input');t.add_argument('output');t.add_argument('--updates',type=int,default=2000)
    q=sub.add_parser('score');q.add_argument('model');q.add_argument('input')
    q=sub.add_parser('gradient');q.add_argument('model');q.add_argument('input')
    q=sub.add_parser('sample');q.add_argument('model');q.add_argument('--count',type=int,default=256);q.add_argument('--seed',type=int,default=19)
    a=p.parse_args()
    try:
        if a.command=='demo':r=demo()
        elif a.command=='train':
            n,h=train(read(a.input),updates=a.updates)
            with open(a.output,'w',encoding='utf-8') as f:json.dump(artifact(n),f,allow_nan=False)
            r={'output':a.output,'history':h}
        elif a.command=='sample':r={'points':load(read(a.model)).sample(a.count,a.seed)}
        else:
            data=read(a.input)
            if not isinstance(data,list) or not 1<=len(data)<=8192:raise ValueError('score input needs 1..8192 points')
            n=load(read(a.model))
            if a.command=='gradient':
                if len(data)>256:raise ValueError('gradient input needs 1..256 points')
                r={'rows':[n.input_score(point(x)) for x in data],
                   'meaning':'gradient of fitted log density in input coordinates; not a probability or OOD detector'}
            else:r={'log_density':[n.log_prob(point(x)) for x in data],'units':'nats per 2-D point'}
        print(json.dumps(r,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError) as e:p.exit(2,f'flowlab: {e}\n')


if __name__=='__main__':main()
