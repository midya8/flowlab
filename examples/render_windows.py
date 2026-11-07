"""Nested-window sensitivity of one sampled interpolant, not tail certification."""
import json
import sys
from pathlib import Path


def render(r):
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="505" viewBox="0 0 960 505" role="img" aria-label="FlowLab conditional window sensitivity">',
       '<rect width="960" height="505" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="32" font-size="23">FlowLab | changing the window changes the distribution</text>',
       '<text x="24" y="57" font-size="13">Same joint-density evaluations. Outer window [-4,4]; aligned inner window [-1,1], re-normalized.</text>']
    for j,row in enumerate(r['window_audit']['rows']):
        ox,oy=40+310*j,105
        p.extend([f'<text x="{ox}" y="86" font-size="16" font-weight="600">Observed x = {row["fixed_value"]:g}</text>',f'<rect x="{ox}" y="{oy}" width="260" height="210" fill="white" stroke="#ddd"/>'])
        for y in [0.,.7,1.4]:
            yy=oy+210-y*150;p.append(f'<path d="M{ox},{yy}h260" stroke="#eae9ea"/><text x="{ox-5}" y="{yy+4}" text-anchor="end" font-size="10">{y:g}</text>')
        for x in [-4,-1,1,4]:p.append(f'<text x="{ox+(x+4)/8*260}" y="{oy+227}" text-anchor="middle" font-size="11">{x}</text>')
        for x in [-1,1]:p.append(f'<path d="M{ox+(x+4)/8*260},{oy}v210" stroke="#ddd" stroke-dasharray="3 3"/>')
        for key,color in [('outer','#585254'),('inner','#387667')]:
            s=row[key];assert max(s['conditional_density'])<1.4
            points=' '.join(f'{ox+(x+4)/8*260:.3f},{oy+210-y*150:.3f}' for x,y in zip(s['grid'],s['conditional_density']))
            p.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        p.append(f'<text x="{ox}" y="363" font-size="12">Removed within outer: {100*row["interpolated_removed_fraction_within_outer"]:.2f}%</text>')
        p.append(f'<text x="{ox}" y="384" font-size="12">Median shift (inner - outer): {row["median_shift_inner_minus_outer"]:+.4f}</text>')
    p.extend(['<text x="24" y="416" font-size="13">Ink: outer-interval density. Teal: inner-interval density, not a crop. 513 evaluations per panel; no re-evaluation.</text>',
              '<text x="24" y="439" font-size="13">Mass ratios describe the interpolant inside the finite outer window. Whole-line tails and missed peaks unknown.</text>',
              '<text x="24" y="462" font-size="13">Adversarial non-neural test: grids17/33 both estimate integral0.0001; analytic narrow-peak integral is0.5001.</text>',
              '<text x="24" y="485" font-size="13">Grid agreement is not an accuracy certificate. No automatic interval selection or calibrated coverage claim.</text>','</g></svg>'])
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    text=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/window-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==text,'stale window-audit.svg'
    else:dest.write_text(text)
