"""ClinVar coding SNVs as codon substitutions, and their zero-shot scores.

The first variant-effect task of the evaluation harness. ClinVar's GRCh38 VCF and NCBI's
``cds_from_genomic`` FASTA are both read with polars-bio. Each coding single-nucleotide variant is placed
on one transcript of its gene (the longest RefSeq CDS that passes :func:`~kron_dna.codons.check_cds`)
through the exon coordinates in the FASTA header (``[location=join(a..b,...)]``, ``complement(...)`` for
the minus strand), and becomes ``(transcript, codon index, reference codon, alternative codon)``. The
consequence (synonymous, missense, nonsense) is recomputed from the codons, and a variant whose
reference base disagrees with the transcript is dropped and counted.

Labels: pathogenic (Pathogenic, Likely pathogenic and both) against benign (Benign, Likely benign and
both), with at least one review star (criteria provided); conflicting and uncertain calls are left out.

The model score is the masked marginal, ``log p(alt codon) - log p(ref codon)`` at the masked codon, in a
window of ``window`` tokens around it; a lower score should mean more damaging, so AUROC is computed for
``-score`` with pathogenic as the positive class. The model-free baselines score the same substitution by
genome-wide codon usage, relative adaptiveness and the change in GC.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
import time

import numpy as np
import polars as pl
import torch

from kron_dna.codons import (CODONS, FIRST_CODON_ID, GENETIC_CODE, MASK_ID, check_cds, clean, encode)
from kron_dna.data import held_out

PATHOGENIC = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
BENIGN = {"Benign", "Likely_benign", "Benign/Likely_benign"}
_REVIEWED = ("criteria_provided", "reviewed_by_expert_panel", "practice_guideline")
_COMPLEMENT = str.maketrans("ACGT", "TGCA")
_GENE = re.compile(r"\[gene=([^\]]+)\]")
_LOCATION = re.compile(r"\[location=([^\]]+)\]")
_ACCESSION = re.compile(r"^(?:lcl\|)?(N[CT]_\d+\.\d+)_cds_")


def chromosome_of(accession: str) -> str | None:
    """RefSeq GRCh38 chromosome accession -> ClinVar's chromosome name (``NC_000001.11`` -> ``1``)."""
    m = re.fullmatch(r"NC_0000(\d\d)\.\d+", accession)
    if m:
        n = int(m.group(1))
        return {23: "X", 24: "Y"}.get(n, str(n)) if 1 <= n <= 24 else None
    return "MT" if accession.startswith("NC_012920.") else None


def parse_location(location: str) -> tuple[str, list[tuple[int, int]]] | None:
    """``join(a..b,c..d)`` / ``complement(...)`` -> ``(strand, [(start, end), ...])``, 1-based closed, in
    genomic order; ``None`` for a partial (``<``/``>``) or otherwise unusual location."""
    if any(ch in location for ch in "<>^") or "order(" in location:
        return None
    strand = "+"
    if location.startswith("complement(") and location.endswith(")"):
        strand, location = "-", location[len("complement("):-1]
    if location.startswith("join(") and location.endswith(")"):
        location = location[len("join("):-1]
    exons = []
    for part in location.split(","):
        m = re.fullmatch(r"(\d+)\.\.(\d+)", part.strip())
        if not m:
            return None
        exons.append((int(m.group(1)), int(m.group(2))))
    return strand, sorted(exons)


def cds_offset(pos: int, strand: str, exons: Sequence[tuple[int, int]]) -> int | None:
    """The 0-based offset into the spliced CDS of genomic position ``pos``, or ``None`` outside its exons."""
    before, total = 0, sum(e - s + 1 for s, e in exons)
    for s, e in exons:
        if s <= pos <= e:
            off = before + pos - s
            return off if strand == "+" else total - 1 - off
        before += e - s + 1
    return None


def read_transcripts(fasta: str | Path) -> dict[str, dict]:
    """``{gene: transcript}``, one per gene: the longest CDS that passes ``check_cds`` and has a usable
    location on a primary chromosome. A transcript holds its name, chromosome, strand, exons and sequence."""
    import polars_bio as pb

    frame = pb.scan_fasta(str(fasta)).collect()
    best: dict[str, dict] = {}
    for name, description, seq in frame.iter_rows():
        gene_m, loc_m, acc_m = _GENE.search(description), _LOCATION.search(description), _ACCESSION.match(name)
        if not (gene_m and loc_m and acc_m):
            continue
        chrom = chromosome_of(acc_m.group(1))
        loc = parse_location(loc_m.group(1))
        seq = clean(seq)
        if chrom is None or loc is None or check_cds(seq) is not None:
            continue
        strand, exons = loc
        if sum(e - s + 1 for s, e in exons) != len(seq):
            continue
        gene = gene_m.group(1)
        if gene not in best or len(seq) > len(best[gene]["seq"]):
            best[gene] = {"name": name, "gene": gene, "chrom": chrom, "strand": strand, "exons": exons, "seq": seq}
    return best


def read_clinvar(vcf: str | Path) -> pl.DataFrame:
    """Reviewed pathogenic and benign SNVs of a ClinVar VCF: chrom, pos (1-based), ref, alt, id, genes, label."""
    import polars_bio as pb

    lf = pb.scan_vcf(str(vcf), info_fields=["CLNSIG", "GENEINFO", "CLNREVSTAT"])
    df = (lf.filter((pl.col("ref").str.len_chars() == 1) & (pl.col("alt").str.len_chars() == 1))
          .select("chrom", "start", "ref", "alt", "id", "CLNSIG", "GENEINFO", "CLNREVSTAT").collect())
    df = df.with_columns(pl.col("CLNSIG").list.join(",").alias("clnsig"),
                         pl.col("CLNREVSTAT").list.join(",").alias("revstat"))
    df = df.filter(pl.col("revstat").str.starts_with(_REVIEWED[0]) | pl.col("revstat").str.starts_with(_REVIEWED[1])
                   | pl.col("revstat").str.starts_with(_REVIEWED[2]))
    df = df.with_columns(pl.when(pl.col("clnsig").is_in(list(PATHOGENIC))).then(1)
                         .when(pl.col("clnsig").is_in(list(BENIGN))).then(0).otherwise(None).alias("label"))
    df = df.filter(pl.col("label").is_not_null() & pl.col("GENEINFO").is_not_null())
    genes = pl.col("GENEINFO").str.split("|").list.eval(pl.element().str.split(":").list.first())
    return df.select(pl.col("chrom"), pl.col("start").alias("pos").cast(pl.Int64), "ref", "alt", "id",
                     genes.alias("genes"), "clnsig", pl.col("label").cast(pl.Int8))


def map_variants(variants: pl.DataFrame, transcripts: dict[str, dict], holdout_genes: float = 0.05,
                 seed: int = 0) -> tuple[pl.DataFrame, Counter]:
    """Place each variant on its gene's transcript as a codon substitution; ``(table, drop counts)``.

    ``in_training`` says whether the gene is outside the ``holdout_genes`` fraction that
    ``prepare --holdout-genes`` set aside with the same seed."""
    rows, dropped = [], Counter()
    for chrom, pos, ref, alt, vid, genes, clnsig, label in variants.iter_rows():
        tx = next((transcripts[g] for g in genes if g in transcripts and transcripts[g]["chrom"] == chrom
                   and cds_offset(pos, transcripts[g]["strand"], transcripts[g]["exons"]) is not None), None)
        if tx is None:
            dropped["not in a chosen CDS"] += 1
            continue
        off = cds_offset(pos, tx["strand"], tx["exons"])
        if tx["strand"] == "-":
            ref, alt = ref.translate(_COMPLEMENT), alt.translate(_COMPLEMENT)
        seq = tx["seq"]
        if seq[off] != ref:
            dropped["reference mismatch"] += 1
            continue
        k = off // 3
        ref_codon = seq[3 * k:3 * k + 3]
        alt_codon = ref_codon[:off % 3] + alt + ref_codon[off % 3 + 1:]
        if alt_codon not in GENETIC_CODE or k == 0 or 3 * k + 3 == len(seq):
            dropped["start/stop codon or unreadable"] += 1
            continue
        a, b = GENETIC_CODE[ref_codon], GENETIC_CODE[alt_codon]
        consequence = "synonymous" if a == b else ("nonsense" if b == "*" else "missense")
        rows.append({"id": vid, "gene": tx["gene"], "transcript": tx["name"], "codon_index": k,
                     "ref_codon": ref_codon, "alt_codon": alt_codon, "consequence": consequence,
                     "clnsig": clnsig, "label": label,
                     "in_training": not held_out(tx["gene"], holdout_genes, seed)})
    return pl.DataFrame(rows), dropped


def codon_frequencies(seqs: Sequence[str]) -> dict[str, float]:
    """Genome-wide codon frequencies (pseudocount 0.5)."""
    counts = Counter(s[i:i + 3] for s in seqs for i in range(0, len(s) - 2, 3))
    total = sum(counts[c] + 0.5 for c in CODONS)
    return {c: (counts[c] + 0.5) / total for c in CODONS}


def baseline_scores(table: pl.DataFrame, freqs: dict[str, float], w: dict[str, float]) -> dict[str, np.ndarray]:
    """Zero-shot model-free scores (higher = more benign-like, as the model's)."""
    ref, alt = table["ref_codon"].to_list(), table["alt_codon"].to_list()
    gc = lambda c: sum(b in "GC" for b in c)  # noqa: E731
    return {"codon_usage_delta": np.array([math.log(freqs[b]) - math.log(freqs[a]) for a, b in zip(ref, alt)]),
            "log_w_delta": np.array([math.log(w[b]) - math.log(w[a]) for a, b in zip(ref, alt)]),
            "gc_delta": np.array([gc(b) - gc(a) for a, b in zip(ref, alt)], dtype=float)}


def sample_per_class(table: pl.DataFrame, n: int, seed: int = 0) -> pl.DataFrame:
    """At most ``n`` variants of each (consequence, label), drawn by seed; ``n <= 0`` keeps all.

    AUROC does not depend on class balance, so capping the benign synonymous majority only costs precision."""
    if n <= 0:
        return table
    parts = [g.sample(min(n, g.height), seed=seed) for _, g in
             table.sort("id").group_by(["consequence", "label"], maintain_order=True)]
    return pl.concat(parts).sort("id")


@torch.no_grad()
def masked_marginals(model: torch.nn.Module, table: pl.DataFrame, transcripts: dict[str, dict],
                     device: torch.device, window: int = 256, batch_size: int = 32, bf16: bool = True) -> np.ndarray:
    """``log p(alt) - log p(ref)`` over the 64 codons at each variant's masked codon, one forward per codon
    position (variants at the same codon share it), in a ``window``-token context centred on the codon."""
    model = model.to(device).eval()
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda")
    by_name = {t["name"]: t for t in transcripts.values()}
    keys = sorted(set(zip(table["transcript"].to_list(), table["codon_index"].to_list())))
    logp: dict[tuple[str, int], np.ndarray] = {}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        rows = []
        for name, k in chunk:
            tokens = encode(by_name[name]["seq"])          # <cls> codons <eos>; codon k is token k + 1
            centre = k + 1
            lo = max(0, min(centre - window // 2, len(tokens) - window))
            piece = list(tokens[lo:lo + window])
            piece[centre - lo] = MASK_ID
            rows.append((piece, centre - lo))
        width = max(len(p) for p, _ in rows)
        batch = torch.zeros((len(rows), width), dtype=torch.long)
        for i, (p, _) in enumerate(rows):
            batch[i, :len(p)] = torch.tensor(p)
        with autocast:
            logits = model(batch.to(device))[0]     # KronDNA or its plain-PyTorch KronDNAGraph (CPU)
        at = torch.tensor([c for _, c in rows], device=device)
        sel = logits[torch.arange(len(rows), device=device), at, FIRST_CODON_ID:].float()
        lp = torch.log_softmax(sel, dim=-1).cpu().numpy()
        for key, row in zip(chunk, lp):
            logp[key] = row
    index = {c: i for i, c in enumerate(CODONS)}
    return np.array([logp[(n, k)][index[b]] - logp[(n, k)][index[a]] for n, k, a, b in
                     zip(table["transcript"].to_list(), table["codon_index"].to_list(),
                         table["ref_codon"].to_list(), table["alt_codon"].to_list())])


@torch.no_grad()
def esm_marginals(table: pl.DataFrame, transcripts: dict[str, dict], device: torch.device,
                  model_name: str = "esm2_t33_650M_UR50D", window: int = 1022, batch_size: int = 8,
                  model=None) -> np.ndarray:
    """ESM-2's masked marginal on the protein, ``log p(alt aa) - log p(ref aa)`` at the masked residue, for the
    missense rows of ``table``; NaN for the rest (a synonymous change leaves the protein as it is).

    The protein is the transcript's translation without its stop; a residue past ``window`` gets a window
    centred on it, as ``masked_marginals`` does for codons. ``model`` is ``(model, alphabet)`` for tests."""
    import esm

    net, alphabet = model if model is not None else getattr(esm.pretrained, model_name)()
    net = net.to(device).eval()                            # fp32: bf16 drops ESM-2 650M from 0.90 to 0.70 AUROC
    by_name = {t["name"]: t for t in transcripts.values()}
    proteins = {name: "".join(GENETIC_CODE[t["seq"][i:i + 3]] for i in range(0, len(t["seq"]) - 3, 3))
                for name, t in by_name.items()}
    missense = table["consequence"].to_numpy() == "missense"
    keys = sorted(set((n, k) for n, k, m in zip(table["transcript"].to_list(), table["codon_index"].to_list(),
                                                 missense) if m), key=lambda nk: (len(proteins[nk[0]]), nk))
    logp: dict[tuple[str, int], np.ndarray] = {}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start:start + batch_size]
        pieces, at = [], []
        for name, k in chunk:
            prot = proteins[name]
            lo = max(0, min(k - window // 2, len(prot) - window))
            pieces.append((name, prot[lo:lo + window]))
            at.append(k - lo + 1)                            # +1: the <cls> token ESM puts first
        _, _, tokens = alphabet.get_batch_converter()(pieces)
        tokens[torch.arange(len(chunk)), torch.tensor(at)] = alphabet.mask_idx
        logits = net(tokens.to(device))["logits"]
        lp = torch.log_softmax(logits[torch.arange(len(chunk), device=device), torch.tensor(at, device=device)]
                               .float(), dim=-1).cpu().numpy()
        for key, row in zip(chunk, lp):
            logp[key] = row
    out = np.full(table.height, np.nan)
    for i, (n, k, a, b, m) in enumerate(zip(table["transcript"].to_list(), table["codon_index"].to_list(),
                                            table["ref_codon"].to_list(), table["alt_codon"].to_list(), missense)):
        if m:
            row = logp[(n, k)]
            out[i] = row[alphabet.get_idx(GENETIC_CODE[b])] - row[alphabet.get_idx(GENETIC_CODE[a])]
    return out


def auroc(score: np.ndarray, label: np.ndarray) -> float | None:
    """AUROC of ``-score`` for ``label == 1`` (pathogenic), by ranks (ties averaged); ``None`` with one class."""
    from scipy.stats import rankdata

    pos, n1 = label == 1, int((label == 1).sum())
    n0 = len(label) - n1
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(-score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def alphamissense_scores(table: pl.DataFrame, positions: pl.DataFrame, path: str | Path) -> np.ndarray:
    """AlphaMissense pathogenicity for each row of ``table``, as a score where lower means pathogenic
    (``-am_pathogenicity``), NaN where AlphaMissense has none (synonymous variants, other transcripts).

    ``positions`` holds ``id, chrom, pos, ref, alt`` (``read_clinvar``) for the table's ids; ``path`` is the
    public ``AlphaMissense_hg38.tsv.gz`` (chromosomes as ``chr1``; several transcripts at one site: the highest)."""
    am = pl.read_csv(path, separator="\t", comment_prefix="#", has_header=False, columns=[0, 1, 2, 3, 8],
                     infer_schema=False)                       # all text; typed below
    am = am.rename(dict(zip(am.columns, ["chrom", "pos", "ref", "alt", "am"])))
    am = (am.with_columns(pl.col("chrom").str.strip_prefix("chr"), pl.col("pos").cast(pl.Int64),
                          pl.col("am").cast(pl.Float64))
          .group_by("chrom", "pos", "ref", "alt").agg(pl.col("am").max()))
    keyed = positions.select("id", pl.col("chrom").cast(pl.Utf8), pl.col("pos").cast(pl.Int64), "ref", "alt")
    joined = table.select("id").join(keyed.join(am, on=["chrom", "pos", "ref", "alt"], how="left"),
                                     on="id", how="left", maintain_order="left")
    return -joined["am"].fill_null(float("nan")).to_numpy().astype(float)


ALPHAGENOME_SCORERS = ("SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS", "RNA_SEQ")


def alphagenome_scores(table: pl.DataFrame, positions: pl.DataFrame, api_key: str, cache: Path,
                       scorers: tuple[str, ...] = ALPHAGENOME_SCORERS, workers: int = 4,
                       client=None, version: int = 1, conditions: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """AlphaGenome's recommended variant scorers over a 100 kb window, one score per scorer: ``-max |effect|``
    over its tracks (and genes, for RNA_SEQ), so lower means a larger predicted effect, as for the other scores.

    ``version`` 2 is AlphaGenome 2, which predicts one condition (tissue, cell type) per call: each variant is scored
    in every one of ``conditions``, and the largest effect over them is kept. Its scores are named ``alphagenome2:``.

    Results are appended to ``cache`` (a parquet keyed by ``id``) as they arrive, so a rerun skips what is done;
    a failed call is retried 5 times, then left NaN and not cached, so a rerun tries it again. ``client`` stands
    in for the API in tests."""
    from concurrent.futures import ThreadPoolExecutor

    from importlib import import_module

    from alphagenome.data import genome

    if version == 2 and not conditions:
        raise ValueError("AlphaGenome 2 needs at least one condition")
    dna_client = import_module(f"alphagenome.models.v{version}.dna_client")
    VS = import_module(f"alphagenome.models.v{version}.variant_scorers")
    if client is None:
        client = dna_client.create(api_key)
    per_call = list(conditions) if version == 2 else [None]

    chosen = [VS.RECOMMENDED_VARIANT_SCORERS[k] for k in scorers]
    done = pl.read_parquet(cache) if cache.exists() else pl.DataFrame(schema={"id": pl.Utf8, **{k: pl.Float64 for k in scorers}})
    keyed = positions.select("id", "chrom", "pos", "ref", "alt").unique("id")
    todo = table.select("id").unique().join(keyed, on="id").join(done.select("id"), on="id", how="anti")

    def one(row: dict) -> dict:
        v = genome.Variant(chromosome=f"chr{row['chrom']}", position=int(row["pos"]),
                           reference_bases=row["ref"], alternate_bases=row["alt"])
        best = {k: float("nan") for k in scorers}
        for condition in per_call:
            extra = {"condition": condition} if condition else {}
            for attempt in range(5):
                try:
                    res = client.score_variant(interval=v.reference_interval.resize(dna_client.SEQUENCE_LENGTH_100KB),
                                               variant=v, variant_scorers=chosen, **extra)
                    break
                except Exception:                             # transient gRPC / proxy failures
                    time.sleep(2 ** attempt)
            else:
                return None
            for k, r in zip(scorers, res):
                if r.X.size:
                    best[k] = float(np.nanmax([best[k], float(np.nanmax(np.abs(r.X)))]))
        return {"id": row["id"], **best}

    rows = todo.to_dicts()
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(rows), 200):
            got = pl.DataFrame([r for r in pool.map(one, rows[start:start + 200]) if r], schema=done.schema)
            done = pl.concat([done, got])
            done.write_parquet(cache)
            print(f"AlphaGenome: {done.height:,} variants scored", flush=True)
    joined = table.select("id").join(done, on="id", how="left", maintain_order="left")
    prefix = "alphagenome" if version == 1 else "alphagenome2"
    return {f"{prefix}:{k.lower()}": -joined[k].fill_null(float("nan")).to_numpy().astype(float) for k in scorers}


def bootstrap_ci(metric, *arrays: np.ndarray, n: int = 1000, seed: int = 0, level: float = 0.95) -> list[float] | None:
    """Percentile interval of ``metric(*arrays)`` over ``n`` resamples of the rows, with replacement.

    Resamples where the metric is undefined (``None``, e.g. one class drawn) are skipped; ``None`` when all are.
    """
    rng = np.random.default_rng(seed)
    rows = len(arrays[0])
    if rows == 0:
        return None
    values = []
    for _ in range(n):
        idx = rng.integers(0, rows, rows)
        v = metric(*(a[idx] for a in arrays))
        if v is not None:
            values.append(v)
    if not values:
        return None
    tail = (1 - level) / 2 * 100
    return [float(np.percentile(values, tail)), float(np.percentile(values, 100 - tail))]


def summarize(table: pl.DataFrame, scores: dict[str, np.ndarray], n_boot: int = 1000) -> list[dict]:
    """AUROC of every score by consequence (synonymous, missense) and gene split (all, in training, held out),
    with a 95% bootstrap interval over the variants as ``<score>_ci`` (``n_boot`` 0 leaves it out)."""
    out = []
    label = table["label"].to_numpy()
    cons = table["consequence"].to_numpy()
    train = table["in_training"].to_numpy()
    for consequence in ("synonymous", "missense"):
        for split, keep in (("all", np.ones(len(table), bool)), ("in_training", train), ("held_out", ~train)):
            m = (cons == consequence) & keep
            row = {"consequence": consequence, "genes": split, "pathogenic": int((label[m] == 1).sum()),
                   "benign": int((label[m] == 0).sum())}
            for name, sc in scores.items():
                ms = m & np.isfinite(sc)                     # a score may cover only part of the variants
                row[name] = auroc(sc[ms], label[ms])
                if ms.sum() < m.sum():
                    row[f"{name}_n"] = int(ms.sum())
                if n_boot:
                    row[f"{name}_ci"] = bootstrap_ci(auroc, sc[ms], label[ms], n=n_boot)
            out.append(row)
    return out
