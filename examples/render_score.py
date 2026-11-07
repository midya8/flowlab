"""Reproduce the measured score decomposition, without taking optimization steps."""
import json
import math
import sys
from pathlib import Path


def render(r):
    a=r['input_score_audit']
    p=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="490" viewBox="0 0 960 490" role="img" aria-label="FlowLab log-density gradient decomposition on a fixed grid">',
       '<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="#387667"/></marker></defs>',
       '<rect width="960" height="490" fill="white"/><g font-family="Arial,sans-serif" fill="#251f21">',
       '<text x="24" y="32" font-size="23">FlowLab | the score needs the volume term</text>',
       '<text x="24" y="57" font-size="13">Same trained flow and 81 grid points. Score = gradient of fitted log density, not a probability.</text>']
    for j,(key,title) in enumerate([('base_pullback','Base-density pullback'),('logdet_gradient','Log-Jacobian gradient'),('score','Full input score')]):
        ox,oy,side=54+310*j,108,240
        p.extend([f'<text x="{ox}" y="87" font-size="16" font-weight="600">{title}</text>',f'<rect x="{ox}" y="{oy}" width="{side}" height="{side}" fill="#fafafa" stroke="#ddd"/>'])
        for v in range(5):
            p.append(f'<path d="M{ox+v*60},{oy}v240 M{ox},{oy+v*60}h240" stroke="#e5e5e5"/>')
        clipped=0;norms=[]
        for row in a['rows']:
            x,y=row['point'];dx,dy=row[key];norm=math.hypot(dx,dy);norms.append(norm)
            factor=.03
            if norm*factor>.32:factor=.32/norm;clipped+=1
            u=ox+(x+2)*60;v=oy+(3-y)*60
            # Boundary points get inward arrows only if their actual vector points inward.
            p.append(f'<circle cx="{u:.3f}" cy="{v:.3f}" r="1.4" fill="#777"/>')
            if norm>1e-12:p.append(f'<path d="M{u:.3f},{v:.3f} l{dx*factor*60:.3f},{-dy*factor*60:.3f}" stroke="#387667" fill="none" stroke-width="1.2" marker-end="url(#arrow)"/>')
        p.extend([f'<text x="{ox}" y="373" font-size="12">Max raw norm: {max(norms):.3f}; capped: {clipped}/81</text>',f'<text x="{ox}" y="393" font-size="12">x: -2 to 2; y: -1 to 3. Shared axes/gain.</text>'])
    p.extend([f'<text x="24" y="424" font-size="13">Central difference, epsilon 1e-5: max coordinate error {a["max_abs_coordinate_error"]:.2e}; omitting volume: {a["max_abs_error_omitting_logdet_gradient"]:.4f}.</text>',
              '<text x="24" y="448" font-size="13">Arrows = 0.03 times vector, length capped at 0.32 input units for display. Raw vectors are in demo JSON.</text>',
              '<text x="24" y="472" font-size="13">Fixed-grid implementation audit, not a global bound, causal explanation, OOD detector or sampling method.</text>','</g></svg>'])
    return '\n'.join(p)+'\n'


if __name__=='__main__':
    text=render(json.loads(Path(sys.argv[1]).read_text()));dest=Path('assets/score-audit.svg')
    if '--check' in sys.argv:assert dest.read_text()==text,'stale score-audit.svg'
    else:dest.write_text(text)
