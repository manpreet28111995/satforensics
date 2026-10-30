import hashlib
import json
import time
from dataclasses import asdict, dataclass, field


@dataclass
class Event:
    timestamp: float
    detection_score: float
    archetype: str
    posterior: list
    top2: list
    margin: float
    entropy: float
    ambiguous: bool
    source: str
    observation: int
    context: dict = field(default_factory=dict)


@dataclass
class Block:
    index: int
    event: Event
    previous_hash: str
    hash: str


class ForensicLedger:
    def __init__(self, genesis=b"satforensics-genesis"):
        self.genesis = hashlib.sha256(genesis).hexdigest()
        self.blocks = []

    @staticmethod
    def encode(event):
        return json.dumps(asdict(event), sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self, previous, event):
        h = hashlib.sha256()
        h.update(previous.encode("utf-8"))
        h.update(self.encode(event))
        h.update(repr(event.timestamp).encode("utf-8"))
        return h.hexdigest()

    def append(self, event):
        previous = self.blocks[-1].hash if self.blocks else self.genesis
        block = Block(len(self.blocks), event, previous, self.digest(previous, event))
        self.blocks.append(block)
        return block

    def verify(self):
        previous = self.genesis
        for i, b in enumerate(self.blocks):
            if b.previous_hash != previous or self.digest(previous, b.event) != b.hash:
                return i
            previous = b.hash
        return -1

    def save(self, path):
        with open(path, "w") as fh:
            for b in self.blocks:
                fh.write(json.dumps({"index": b.index, "previous_hash": b.previous_hash, "hash": b.hash,
                                     "event": asdict(b.event)}) + "\n")


def make_event(i, rate_scale=1.0):
    post = [0.7, 0.1, 0.1, 0.05, 0.05]
    return Event(time.time(), 0.97, "BJ", post, ["BJ", "PB"], 0.6, 0.31, False, "station-%d" % (i % 50), i,
                 {"mcs": "GFSK9k6", "csi_age_ms": 7.5, "retransmissions": 0})


def benchmark(n=20000, tamper_trials=200, seed=0):
    import random
    import statistics
    rng = random.Random(seed)
    led = ForensicLedger()
    lat = []
    for i in range(n):
        ev = make_event(i)
        t = time.perf_counter()
        led.append(ev)
        lat.append((time.perf_counter() - t) * 1000.0)
    t = time.perf_counter()
    ok = led.verify()
    verify_ms = (time.perf_counter() - t) * 1000.0
    base = []
    sink = []
    for i in range(n):
        ev = make_event(i)
        t = time.perf_counter()
        sink.append(ForensicLedger.encode(ev))
        base.append((time.perf_counter() - t) * 1000.0)
    detected = 0
    for _ in range(tamper_trials):
        k = rng.randrange(len(led.blocks))
        saved = led.blocks[k].event.margin
        led.blocks[k].event.margin = saved + 0.01
        if led.verify() == k:
            detected += 1
        led.blocks[k].event.margin = saved
    lat.sort()
    base.sort()
    return {
        "n": n,
        "median_ms": statistics.median(lat),
        "p95_ms": lat[int(0.95 * len(lat))],
        "max_throughput_eps": 1000.0 / statistics.mean(lat),
        "baseline_median_ms": statistics.median(base),
        "verify_all_ms": verify_ms,
        "verify_ok": ok == -1,
        "tamper_detected": detected,
        "tamper_trials": tamper_trials,
    }
