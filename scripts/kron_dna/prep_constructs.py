"""Annotated engineered constructs (plasmids, vectors) from GenBank flat files, as a small prepared genome.

    uv run python scripts/kron_dna/prep_constructs.py OUT_DIR GBFF[.gz] [GBFF ...] [--min-len 2000] [--max-len 50000]

Any GenBank flat file serves: the synthetic-construct division of GenBank (``gbsyn*.seq.gz``), a vendor's vector
set, a depositor's export. A record is kept when its length is in range, it has no base outside ACGT beyond
``--max-n`` of its length, it carries at least one feature of ``PART_BITS`` and its sequence was not seen before
(the same vector is deposited many times; sequences are compared exactly, so a backbone with another insert stays).

OUT_DIR, in ``genome-prepare``'s layout (one contig per record, laid end to end):

- ``bases.u8`` and ``contigs.parquet`` (chrom = accession, start 0, end, offset, circular, organism, definition,
  held_out: one record in ``--held-out-mod`` by the MD5 of its sequence, for scoring annotation on unseen constructs);
- ``parts.u8``: per base the bits of ``PART_BITS``. A feature's strand is kept only for coding sequence
  (``CDS_plus`` / ``CDS_minus``); a feature that runs through the origin of a circular record paints both ends;
- ``labels.u8``: the genome label byte (``genome.LABEL_BITS``) with ``promoter``, ``enhancer`` and ``CDS`` set from
  the same features and every other bit zero. A construct's genes are mostly intronless cDNA: the zero ``exon``,
  ``donor`` and ``acceptor`` bits mean "not annotated", and a loss on them should be switched off for these rows;
- ``constructs.json``: counts, per part the share of bases, what was left out and why.

Feature keys read: ``CDS``, ``intron``, ``polyA_signal``, ``rep_origin``, ``terminator``, the old keys
``promoter`` and ``enhancer``, and ``regulatory`` with ``/regulatory_class`` promoter, enhancer, terminator or
polyA_signal_sequence. Locations with a reference to another record are skipped.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import polars as pl
import typer

from kron_dna.genome import LABEL_BITS, encode_bases

PART_BITS = {"promoter": 0, "enhancer": 1, "CDS_plus": 2, "CDS_minus": 3, "polyA_signal": 4, "intron": 5,
             "rep_origin": 6, "terminator": 7}
KEYS = {"promoter": "promoter", "enhancer": "enhancer", "polyA_signal": "polyA_signal", "intron": "intron",
        "rep_origin": "rep_origin", "terminator": "terminator", "CDS": "CDS"}
CLASSES = {"promoter": "promoter", "enhancer": "enhancer", "terminator": "terminator",
           "polyA_signal_sequence": "polyA_signal"}
SPAN = re.compile(r"[<>]?(\d+)(?:\.\.[<>]?(\d+))?")


def spans(location: str) -> tuple[list[tuple[int, int]], bool]:
    """A feature location as 0-based half-open spans and whether it is on the complement strand; no spans for a
    location that points into another record (``J00194.1:100..202``) or between bases (``123^124``)."""
    if ":" in location or "^" in location:
        return [], False
    out = [(int(a) - 1, int(b or a)) for a, b in SPAN.findall(location)]
    return [(a, b) for a, b in out if b > a], location.lstrip().startswith("complement")


def records(lines):
    """Records of a GenBank flat file: dicts of accession, definition, organism, circular, sequence (upper case)
    and features ``[(part, spans, complement)]`` for the parts of ``PART_BITS`` (CDS before its strand is applied)."""
    rec, section, feat = None, None, None

    def close_feature():
        if feat and feat["part"]:
            sp, comp = spans(feat["loc"])
            if sp:
                rec["features"].append((feat["part"], sp, comp))

    for line in lines:
        if line.startswith("LOCUS"):
            rec = {"accession": line.split()[1], "definition": "", "organism": "", "features": [],
                   "circular": " circular " in line}
            section, feat, seq = None, None, []
        elif rec is None:
            continue
        elif line.startswith("DEFINITION"):
            rec["definition"], section = line[12:].strip(), "definition"
        elif section == "definition" and line.startswith(" " * 12):
            rec["definition"] += " " + line.strip()
        elif line.startswith("VERSION"):
            rec["accession"], section = (line.split() + [rec["accession"]])[1], None
        elif line.startswith("  ORGANISM"):
            rec["organism"], section = line[12:].strip(), None
        elif line.startswith("FEATURES"):
            section = "features"
        elif line.startswith("ORIGIN"):
            close_feature()
            section, feat = "origin", None
        elif line.startswith("//"):
            close_feature()
            rec["sequence"] = "".join(seq).upper()
            yield rec
            rec, section, feat = None, None, None
        elif section == "origin":
            seq.append("".join(line.split()[1:]))
        elif section == "features" and line.startswith("     ") and line[5:6] != " ":
            close_feature()
            key, loc = line[5:21].strip(), line[21:].strip()
            feat = {"key": key, "loc": loc, "part": KEYS.get(key), "in_loc": True}
        elif section == "features" and feat is not None and line.startswith(" " * 21):
            s = line[21:].strip()
            if s.startswith("/"):
                feat["in_loc"] = False
                if feat["key"] == "regulatory" and s.startswith("/regulatory_class="):
                    feat["part"] = CLASSES.get(s.split("=", 1)[1].strip('"'))
            elif feat["in_loc"]:
                feat["loc"] += s                                        # a location continued on the next line
        elif line[:1] != " ":
            section = None


def paint(rec: dict) -> np.ndarray:
    """The record's ``parts`` byte per base."""
    n = len(rec["sequence"])
    out = np.zeros(n, dtype=np.uint8)
    for part, sp, comp in rec["features"]:
        bit = PART_BITS[("CDS_minus" if comp else "CDS_plus") if part == "CDS" else part]
        for a, b in sp:
            out[a:min(b, n)] |= np.uint8(1 << bit)
    return out


def to_labels(parts: np.ndarray) -> np.ndarray:
    """``parts`` as the genome label byte: promoter, enhancer and CDS; the other bits zero (not annotated)."""
    out = np.zeros(len(parts), dtype=np.uint8)
    for name, bits in (("promoter", ["promoter"]), ("enhancer", ["enhancer"]), ("CDS", ["CDS_plus", "CDS_minus"])):
        m = np.zeros(len(parts), dtype=bool)
        for b in bits:
            m |= ((parts >> PART_BITS[b]) & 1).astype(bool)
        out[m] |= np.uint8(1 << LABEL_BITS[name])
    return out


def main(out: Path, gbff: list[Path], min_len: int = 2000, max_len: int = 50000, max_n: float = 0.01,
         held_out_mod: int = 10) -> None:
    left = dict(short=0, long=0, ambiguous=0, no_parts=0, duplicate=0, no_sequence=0)
    seen: set[bytes] = set()
    rows, base_chunks, part_chunks, offset = [], [], [], 0
    for path in gbff:
        with (gzip.open(path, "rt", errors="replace") if path.suffix == ".gz" else open(path, errors="replace")) as fh:
            for r in records(fh):
                n = len(r["sequence"])
                if n == 0:
                    left["no_sequence"] += 1
                    continue
                if n < min_len or n > max_len:
                    left["short" if n < min_len else "long"] += 1
                    continue
                if not r["features"]:
                    left["no_parts"] += 1
                    continue
                codes = encode_bases(r["sequence"])
                if float((codes > 3).mean()) > max_n:
                    left["ambiguous"] += 1
                    continue
                key = hashlib.md5(r["sequence"].encode()).digest()
                if key in seen:
                    left["duplicate"] += 1
                    continue
                seen.add(key)
                base_chunks.append(codes)
                part_chunks.append(paint(r))
                rows.append({"chrom": r["accession"], "start": 0, "end": n, "offset": offset, "circular": r["circular"],
                             "organism": r["organism"], "definition": r["definition"][:300],
                             "held_out": int.from_bytes(key[:8], "big") % held_out_mod == 0, "source": path.name})
                offset += n
    if not rows:
        raise typer.BadParameter(f"no record kept; left out: {left}")
    out.mkdir(parents=True, exist_ok=True)
    parts = np.concatenate(part_chunks)
    np.concatenate(base_chunks).tofile(out / "bases.u8")
    parts.tofile(out / "parts.u8")
    to_labels(parts).tofile(out / "labels.u8")
    table = pl.DataFrame(rows)
    table.write_parquet(out / "contigs.parquet")
    info = dict(records=len(rows), bases=offset, held_out=int(table["held_out"].sum()), circular=int(table["circular"].sum()),
                left_out=left, part_bits=PART_BITS,
                share_of_bases={p: round(float(((parts >> b) & 1).mean()), 5) for p, b in PART_BITS.items()},
                records_with={p: int(sum(bool(((c >> b) & 1).any()) for c in part_chunks)) for p, b in PART_BITS.items()},
                sources=[p.name for p in gbff], min_len=min_len, max_len=max_len,
                labels="labels.u8 carries promoter, enhancer and CDS only; exon, donor and acceptor are not annotated")
    (out / "constructs.json").write_text(json.dumps(info, indent=1))
    typer.echo(json.dumps(info))


if __name__ == "__main__":
    typer.run(main)
