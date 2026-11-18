"""Is the split-against-whole gradient difference upstream of the core the new head's, the split path's, or the
core kernels' own round-off? Four comparisons on one tiny model, real max-pools, per parameter group:
  A  new head's loss:        whole against split                       (the failing test)
  B  inherited NTv3 head:    whole against split                       (the path the running arms already train on)
  C  new head's loss:        whole against the same whole pass again   (are the kernels deterministic?)
  D  new head's loss:        whole against whole with the stem weights scaled by 1 + 2e-7
                                                                       (how much a round-off-sized change is amplified)
"""
import os, sys, tempfile
os.environ["TRITON_F32_DEFAULT"] = "ieee"
from pathlib import Path
import torch
sys.path.insert(0, "tests")                          # run from the repository root, on a CUDA device
import test_kron_dna_split as TS
import test_kron_dna_xheads_stage as XS
from kron_dna import genome as G, xheads as X

dev = torch.device("cuda")
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
real = G.KronGenome.split_replica
G.KronGenome.split_replica = lambda self, d, force=False: real(self, d, force=True)


def grads(m):
    out = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    return out


def report(tag, g0, g1):
    worst = {}
    for n, g in g0.items():
        d = (g1[n] - g).abs()
        key = n.split(".")[0]
        rel = float(d.max()) / (float(g.abs().max()) + 1e-12)
        worst[key] = max(worst.get(key, 0.0), rel)
    up = max(worst.get(k, 0.0) for k in ("stem", "down", "core", "context", "stem_res"))
    down = max(v for k, v in worst.items() if k not in ("stem", "down", "core", "context", "stem_res"))
    print(f"  {tag}: largest |diff| / max|grad|  upstream of the core {up:.2e}   decoder and heads {down:.2e}")


for by, b, length in (("length", 2, 2048), (None, 4, 1024)):
    tmp = Path(tempfile.mkdtemp())
    XS.write_store(tmp / "s", total=b * length)
    st = X.TrackStore.open(tmp / "s")
    torch.manual_seed(2)
    m = G.KronGenome(G.KronGenomeConfig(**(TS.small("v2").config.to_dict() | {"xheads": [["meas", 3]]}))).to(dev).eval()
    seq, seg, ctx, extra = TS.batch_for(b, length, dev)
    want = torch.stack([torch.from_numpy(st.window(r * length, length, length)) for r in range(b)])
    split = torch.ones(b, length, dtype=torch.long); split[0, :100] = 4
    ok = X.supervised(split, center=0.75)
    g = torch.Generator().manual_seed(9)
    nt_want = (torch.rand(b, length, 5, generator=g) * 3).to(dev)
    nt_mask = (torch.rand(b, length, 1, generator=g) > 0.3).float().to(dev)
    print(f"by={by} b={b} L={length}")

    def whole_x():
        m(seq, seg, ctx, extra)
        X.head_loss(m.last_x["meas"], want.to(dev), ok.to(dev), st.consts(dev)).backward()
        return grads(m)

    def split_x():
        pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[dev, dev], by=by)
        X.split_head_loss(pieces, "meas", want, ok, st.consts, dev).backward()
        return grads(m)

    def whole_nt():
        m(seq, seg, ctx, extra)
        G.poisson_multinomial_loss(torch.nn.functional.softplus(m.last_ntv3.float()), nt_want, nt_mask).backward()
        return grads(m)

    def split_nt():
        pieces, _ = m.forward_split(seq, seg, ctx, extra, devices=[dev, dev], by=by)
        G.poisson_multinomial_loss(torch.nn.functional.softplus(G.gather_split(pieces, "ntv3", dev).float()), nt_want,
                                   nt_mask).backward()
        return grads(m)

    a = whole_x()
    report("A new head, whole vs split      ", a, split_x())
    report("B NTv3 head, whole vs split     ", whole_nt(), split_nt())
    report("C new head, whole vs whole again", a, whole_x())
    with torch.no_grad():
        m.stem.weight.mul_(1 + 2e-7)
    report("D new head, whole vs stem*(1+2e-7)", a, whole_x())
