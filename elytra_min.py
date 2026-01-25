from math import pi, cos, sin
def next_v(v, a):
    assert -pi / 2 < a < pi / 2
    vx, vy = v
    assert vx >= 0
    c = cos(a); c2 = c * c
    vx0 = vx; vy += 0.06 * c2 - 0.08
    if vy < 0:
        t0 = 0.1 * vy * c2
        vx -= t0; vy -= t0
    if a < 0:
        t1 = 0.04 * vx0 * sin(a)
        vx += t1; vy -= 3.2 * t1
    vx += (vx0 - vx) * 0.1
    return 0.99 * vx, 0.98 * vy
