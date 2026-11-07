"""Actual flow/reference/control samples on shared axes, with outlier counts."""
import json
import sys
from pathlib import Path


def render(r):
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="515" viewBox="0 0 960 515" role="img" aria-label="FlowLab learned density and shared-axis sample audit">',
       '<rect width="960" height="515" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="34" font-size="24">FlowLab | exact model density, imperfect fit</text>',
       '<text x="24" y="59" font-size="13">Banana-shaped synthetic data. Same axes: x from -3 to 3, y from -2 to 5. 256 samples per panel.</text>']
    for j,(key,title) in enumerate([('reference','Reference'),('trained_flow','Trained flow'),('fitted_gaussian','Fitted Gaussian'),('untrained_flow','Untrained flow')]):
        ox,oy=24+234*j,101;side=210
        p += [f'<text x="{ox}" y="87" font-size="15" font-weight="600">{title}</text>',
              f'<rect x="{ox}" y="{oy}" width="210" height="210" rx="8" fill="white" stroke="#eae9ea"/>']
        for u in [-2,0,2]:
            p.append(f'<path d="M{ox+(u+3)/6*side},{oy}v210" stroke="#eae9ea"/>')
        points=[]
        for i in range(101):
            x=-3+6*i/100;y=.6*(x*x-1)
            points.append(f'{ox+(x+3)/6*side:.2f},{oy+(5-y)/7*side:.2f}')
        p.append('<polyline points="'+' '.join(points)+'" fill="none" stroke="#c0bebf"/>')
        outside=0
        for x,y in r['samples'][key]:
            if not (-3<=x<=3 and -2<=y<=5):outside+=1;continue
            color='#73a89a' if key=='trained_flow' else '#585254'
            p.append(f'<circle cx="{ox+(x+3)/6*side:.3f}" cy="{oy+(5-y)/7*side:.3f}" r="2.2" fill="{color}" opacity=".6"/>')
        m=r['sample_summary'][key]
        lines=[f'Outside plot: {outside}/256',f'Residual second moment: {m["curve_residual_second_moment"]:.4f}',f'Within residual +/-1: {100*m["fraction_abs_curve_residual_le_1"]:.1f}%']
        if key!='reference':lines.append(f'Test NLL: {r["nll_nats_per_2d_point"][key]["test"]:.4f} nats')
        for i,line in enumerate(lines):p.append(f'<text x="{ox}" y="{333+22*i}" font-size="12">{line}</text>')
    p += ['<text x="24" y="440" font-size="13">Test generating-reference NLL: 2.1514 nats/2-D point. Trained flow: 2.1884. Exact density is not a perfect fit.</text>',
          '<text x="24" y="464" font-size="13">Curve residual = y - 0.6(x squared - 1). These descriptive sample summaries are not calibrated coverage.</text>',
          '<text x="24" y="488" font-size="13">One development fixture, no setting/seed sweep or real-data benchmark. Curve line is the known generator mean.</text>',
          '</g></svg>']
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    svg=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/flow-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==svg,'stale flow-audit.svg'
    else:dest.write_text(svg)
