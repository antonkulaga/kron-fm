"""How well a codon head predicts held-out codons' protein features, beside what the amino acid alone gives.

    python scripts/kron_dna/codon_eval.py CKPT CODON_STORE OUT.json [--windows 64] [--context 1048576] [--name hg38]
        [--mask-share 0.15] [--seed 0] [--centre]

The codon store (prep_esm_atlas_codons.py) gives per codon a target direction (``genome.protein_targets``). Windows
of ``--context`` bases are drawn around held-out codons (``ntv3/split.u8`` != 1 at the codon's first base), each
inside one contig. Per window two forwards: the clean sequence, and the same with ``--mask-share`` of its codons
hidden (all three bases MASK), which is how the head was trained (``--codon-on masked``). Cosine between the head's
output and the target (higher is better), over the held-out codons of the windows:

- ``head_clean``: the base is visible; ``head_masked``: the codon is hidden, so the amino acid must be inferred;
- ``aa_mean``: the mean target of the codon's amino acid over training codons, the score of a head that knows the
  amino acid and nothing of its context (for a hidden codon it is an upper bound on that);
- ``global_mean``: the mean target of all training codons.

``--centre``: targets with the training codons' mean taken out (``genome.protein_centre`` over the script's sample
of training codons), for a head trained on centred targets; without it the targets are as the first arm had them,
where a constant already scores 0.48.

``head_clean`` above ``aa_mean`` means the head reads protein context beyond the residue's identity. OUT.json also
has the scores per amino acid.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from kron_core.paths import interim_dir
from kron_dna import genome as G

AA = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"       # codon index 16a + 4b + c, A C G T


def amino_acids(bases: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Amino-acid index (0..63 -> letter via AA) of codons at layout offsets ``pos`` ``[n, 3]`` in reading order;
    descending offsets are the minus strand, read as the complement. -1 where a base is not A, C, G or T."""
    c = np.asarray(bases[pos.reshape(-1)]).reshape(-1, 3).astype(np.int64)
    minus = pos[:, 0] > pos[:, 2]
    bad = (c > 3).any(1)
    c = np.where(minus[:, None], 3 - c, c)
    idx = 16 * c[:, 0] + 4 * c[:, 1] + c[:, 2]
    letters = np.frombuffer(AA.encode(), dtype=np.uint8)
    return np.where(bad, -1, letters[np.clip(idx, 0, 63)].astype(np.int64))


def class_means(targets: torch.Tensor, aa: np.ndarray) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Unit mean target per amino acid and over all codons."""
    means = {int(a): F.normalize(targets[torch.from_numpy(aa == a)].mean(0), dim=0) for a in np.unique(aa) if a >= 0}
    return means, F.normalize(targets.mean(0), dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"), ap.add_argument("store", type=Path), ap.add_argument("out", type=Path)
    ap.add_argument("--windows", type=int, default=64), ap.add_argument("--context", type=int, default=1 << 20)
    ap.add_argument("--name", default="hg38"), ap.add_argument("--mask-share", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0), ap.add_argument("--train-sample", type=int, default=300_000)
    ap.add_argument("--centre", action="store_true")
    a = ap.parse_args()
    genome = interim_dir("genome", a.name)
    info = json.loads((a.store / "codons.json").read_text())
    contigs = pl.read_parquet(genome / "contigs.parquet").sort("offset")
    if info.get("layout_sha256") != G.layout_fingerprint(*(contigs.sort("chrom", "start")[c] for c in ("chrom", "start", "end", "offset"))):
        raise SystemExit("the codon store was built on another genome layout")
    n = info["codons"]
    pos = np.memmap(a.store / "pos.i64", dtype=np.int64, mode="r", shape=(n, 3))
    feat = np.memmap(a.store / "feat.u16", dtype=np.uint16, mode="r", shape=(n, 64))
    val = np.memmap(a.store / "val.u8", dtype=np.uint8, mode="r", shape=(n, 64))
    scale = np.memmap(a.store / "scale.f32", dtype=np.float32, mode="r", shape=(n,))
    bases = np.memmap(genome / "bases.u8", dtype=np.uint8, mode="r")
    split = np.memmap(genome / "ntv3" / "split.u8", dtype=np.uint8, mode="r")
    first = np.asarray(pos[:, 0])
    held = np.asarray(split[first]) != 1
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda")
    torch.cuda.init()
    model, _ = G.load_checkpoint(a.ckpt)
    model = model.to(dev).eval()
    dims = model.config.codon_dims
    if not dims:
        raise SystemExit("the checkpoint has no codon head")

    centre = None

    def parts(rows: np.ndarray) -> tuple:
        rows = np.sort(rows)
        return (torch.from_numpy(np.asarray(feat[rows]).astype(np.int64)).to(dev),
                torch.from_numpy(np.asarray(val[rows])).to(dev), torch.from_numpy(np.asarray(scale[rows])).to(dev))

    def targets(rows: np.ndarray) -> torch.Tensor:
        return G.protein_targets(*parts(rows), dims, centre)

    train_rows = np.sort(rng.choice(np.flatnonzero(~held), min(a.train_sample, int((~held).sum())), replace=False))
    if a.centre:
        centre = G.protein_centre(*parts(train_rows), dims)
    means, overall = class_means(targets(train_rows), amino_acids(bases, np.asarray(pos[train_rows])))
    starts, ends = contigs["offset"].to_numpy(), (contigs["offset"] + contigs["end"] - contigs["start"]).to_numpy()
    unit, ctx = 2 ** model.config.n_down, a.context
    acc: dict[str, list] = {k: [] for k in ("head_clean", "head_masked", "aa_mean", "aa_mean_masked", "global_mean", "aa")}
    done, tries, seen = 0, 0, np.zeros(n, dtype=bool)
    cand = np.flatnonzero(held)
    while done < a.windows and tries < 20 * a.windows:
        tries += 1
        centre = int(first[rng.choice(cand)])
        k = int(np.searchsorted(starts, centre, side="right") - 1)
        if ends[k] - starts[k] < ctx:
            continue
        lo = min(max(centre - ctx // 2, starts[k]), ends[k] - ctx)
        lo = starts[k] + (lo - starts[k]) // unit * unit
        i0, i1 = np.searchsorted(first, [lo - 2_000_000, lo + ctx + 2_000_000])
        rows = np.arange(i0, i1)
        p = np.asarray(pos[i0:i1])
        rows = rows[((p >= lo) & (p < lo + ctx)).all(1) & held[i0:i1] & ~seen[i0:i1]]
        if len(rows) < 50:
            continue
        seen[rows] = True
        p = np.asarray(pos[rows]) - lo
        aa = amino_acids(bases, np.asarray(pos[rows]))
        want = targets(rows)
        codes = torch.from_numpy(np.asarray(bases[lo:lo + ctx]).astype(np.int64)).to(dev)
        hidden = rng.random(len(rows)) < a.mask_share
        masked = codes.clone()
        masked[torch.from_numpy(p[hidden].reshape(-1)).to(dev)] = G.MASK_CHANNEL
        seg = torch.zeros(1, ctx // unit, dtype=torch.long, device=dev)
        cid = torch.zeros(1, dtype=torch.long, device=dev)
        outs = []
        for x in (codes, masked):
            model.codon_at = (torch.zeros(len(rows), dtype=torch.long), torch.from_numpy(p))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x[None], seg, cid)
            outs.append(F.cosine_similarity(model.last_codon.float(), want, dim=-1).cpu().numpy())
        base = torch.stack([means.get(int(x), overall) for x in aa]).to(dev)
        aa_cos = F.cosine_similarity(base, want, dim=-1).cpu().numpy()
        acc["head_clean"].append(outs[0]), acc["head_masked"].append(outs[1][hidden])
        acc["aa_mean"].append(aa_cos), acc["aa_mean_masked"].append(aa_cos[hidden]), acc["aa"].append(aa)
        acc["global_mean"].append(F.cosine_similarity(overall[None].to(dev), want, dim=-1).cpu().numpy())
        done += 1
        print(f"window {done}: {len(rows)} held-out codons, head {outs[0].mean():.3f}, amino-acid mean {aa_cos.mean():.3f}", flush=True)
    cat = {k: np.concatenate(v) for k, v in acc.items()}
    per_aa = {chr(int(x)): {"n": int((cat["aa"] == x).sum()), "head_clean": float(cat["head_clean"][cat["aa"] == x].mean()),
                            "aa_mean": float(cat["aa_mean"][cat["aa"] == x].mean())} for x in np.unique(cat["aa"]) if x >= 0}
    res = {"checkpoint": str(a.ckpt), "store": a.store.name, "windows": done, "context": ctx, "codons": int(len(cat["aa"])),
           "hidden_codons": int(len(cat["head_masked"])), "mask_share": a.mask_share, "centred": bool(a.centre), "metric": "cosine to the target, higher is better",
           **{k: float(cat[k].mean()) for k in ("head_clean", "head_masked", "aa_mean", "aa_mean_masked", "global_mean")},
           "per_amino_acid": per_aa}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_amino_acid"}, indent=1))


if __name__ == "__main__":
    main()
