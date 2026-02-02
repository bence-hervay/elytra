def next_vx_vy(vx, vy, s):
    assert vx >= 0 and -1 <= s <= 1
    c2 = 1 - s * s
    vx0 = vx
    vy += 0.06 * c2 - 0.08
    if vy < 0:
        ty = 0.1 * vy * c2
        vx -= ty
        vy -= ty
    if s > 0:
        tx = 0.04 * vx0 * s
        vx -= tx
        vy += 3.2 * tx
    vx += 0.1 * (vx0 - vx)
    return 0.99 * vx, 0.98 * vy
