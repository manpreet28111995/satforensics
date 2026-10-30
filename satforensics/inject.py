import numpy as np

ARCHETYPES = ["BJ", "CW", "SW", "PB", "SP"]
DB_LO = -115.0
DB_SPAN = 60.0
BAND_LO = 12
BAND_HI = 116
F_BINS = 128


def to_power(level):
    db = DB_LO + DB_SPAN * level / 255.0
    return np.power(10.0, db / 10.0)


def to_level(power):
    db = 10.0 * np.log10(np.maximum(power, 1e-30))
    return np.clip((db - DB_LO) / DB_SPAN, 0.0, 1.0) * 255.0


def noise_floor(power):
    return np.median(power[:, BAND_LO:BAND_HI])


def fading(rng, shape):
    return rng.exponential(1.0, size=shape)


def pattern_bj(rng, t, f):
    shape = np.zeros((t, f))
    shape[:, BAND_LO:BAND_HI] = fading(rng, (t, BAND_HI - BAND_LO))
    return shape


def pattern_cw(rng, t, f):
    shape = np.zeros((t, f))
    f0 = rng.integers(BAND_LO + 8, BAND_HI - 8)
    width = rng.integers(1, 3)
    amp = 1.0 + 0.1 * rng.standard_normal((t, 1))
    shape[:, f0 - width:f0 + width + 1] = np.maximum(amp, 0.2)
    return shape


def pattern_sw(rng, t, f):
    shape = np.zeros((t, f))
    period = int(rng.integers(5, 13))
    width = int(rng.integers(2, 5))
    phase = rng.uniform(0, 1)
    span = BAND_HI - BAND_LO
    for row in range(t):
        frac = ((row / period) + phase) % 1.0
        centre = int(BAND_LO + frac * span)
        lo, hi = max(centre - width, 0), min(centre + width + 1, f)
        shape[row, lo:hi] = 1.0
    return shape * fading(rng, (t, f)) ** 0.3


def pattern_pb(rng, t, f):
    shape = np.zeros((t, f))
    duty = rng.uniform(0.12, 0.35)
    row = int(rng.integers(0, 4))
    while row < t:
        length = int(rng.integers(1, 4))
        shape[row:row + length, BAND_LO:BAND_HI] = fading(rng, (min(length, t - row), BAND_HI - BAND_LO))
        row += int(max(length / duty, length + 1) * rng.uniform(0.7, 1.3))
    return shape


def pattern_sp(rng, t, f, doppler_shape):
    shape = np.zeros((t, f))
    span = rng.uniform(10.0, 36.0)
    offset = rng.choice([-1.0, 1.0]) * rng.uniform(8.0, 26.0)
    centre = 64.0 + offset
    width = int(rng.integers(1, 3))
    start = int(rng.integers(0, t // 4))
    stop = int(rng.integers(3 * t // 4, t + 1))
    for row in range(start, stop):
        pos = int(round(centre + span * doppler_shape[row]))
        if BAND_LO + width <= pos < BAND_HI - width:
            shape[row, pos - width:pos + width + 1] = 1.0
    return shape


def apply(level, archetype, jnr_db, rng, doppler_shape):
    t, f = level.shape
    power = to_power(level)
    floor = noise_floor(power)
    builders = {"BJ": pattern_bj, "CW": pattern_cw, "SW": pattern_sw, "PB": pattern_pb}
    if archetype == "SP":
        shape = pattern_sp(rng, t, f, doppler_shape)
    else:
        shape = builders[archetype](rng, t, f)
    extra = floor * (10.0 ** (jnr_db / 10.0)) * shape
    return to_level(power + extra)


def draw_jnr(rng, lo=2.0, hi=18.0):
    return rng.uniform(lo, hi)
