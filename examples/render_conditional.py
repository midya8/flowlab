"""Conditional density slices on shared axes; full-line references marked."""
import json
import math
import sys
from pathlib import Path


def render(r):
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="490" viewBox="0 0 960 490" role="img" aria-label="FlowLab finite-interval conditional density slices">',
       '<rect width="960" height="490" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="32" font-size="23">FlowLab | condition on one coordinate, expose the grid</text>',
       '<text x="24" y="57" font-size="13">Flow: piecewise-linear density renormalized on y in [-4,5]. Generator and Gaussian: full-line formulas.</text>']
    def curve(xs,ys,ox,oy,color,dash='none'):
        points=' '.join(f'{ox+(x+4)/9*260:.3f},{oy+220-y*220:.3f}' for x,y in zip(xs,ys))
        return f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>'
    for j,row in enumerate(r['conditional_audit']['rows']):
        ox,oy=40+310*j,105;s=row['slices'][-1];xs=s['grid']
        p.extend([f'<text x="{ox}" y="86" font-size="16" font-weight="600">Observed x = {row["fixed_x"]:g}</text>',f'<rect x="{ox}" y="{oy}" width="260" height="220" fill="white" stroke="#ddd"/>'])
        for y in [0.,.5,1.]:
            yy=oy+220-y*220;p.append(f'<path d="M{ox},{yy}h260" stroke="#eae9ea"/><text x="{ox-5}" y="{yy+4}" text-anchor="end" font-size="10">{y:g}</text>')
        for x in [-4,0,5]:p.append(f'<text x="{ox+(x+4)/9*260}" y="{oy+237}" text-anchor="middle" font-size="11">{x}</text>')
        p.append(curve(xs,s['conditional_density'],ox,oy,'#387667'))
        for mean,sd,color,dash in [(row['generating_mean'],row['generating_sd'],'#585254','5 3'),(row['gaussian_conditional_mean'],row['gaussian_conditional_sd'],'#aaa','2 3')]:
            p.append(curve(xs,[math.exp(-.5*((x-mean)/sd)**2)/(sd*math.sqrt(2*math.pi)) for x in xs],ox,oy,color,dash))
        q=[v['value'] for v in s['quantiles']]
        p.append(f'<text x="{ox}" y="370" font-size="12">Flow median: {q[1]:.4f}; variance: {s["variance"]:.4f}</text>')
        p.append(f'<text x="{ox}" y="391" font-size="12">Flow 5%-95% quantiles: [{q[0]:.4f}, {q[2]:.4f}]</text>')
    p.extend(['<text x="24" y="420" font-size="13">Solid teal: flow (513 grid points). Dashed ink: known generator. Dotted gray: train-fitted Gaussian.</text>',
              '<text x="24" y="443" font-size="13">Shared y range [-4,5], density range [0,1]. Grids 65/257/513 retained in JSON; 2,505 evaluations total.</text>',
              '<text x="24" y="467" font-size="13">Median change 257 to 513 &lt;0.002 in demo. Not a tail-mass check, accuracy bound or calibrated coverage.</text>','</g></svg>'])
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    text=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/conditional-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==text,'stale conditional-audit.svg'
    else:dest.write_text(text)
