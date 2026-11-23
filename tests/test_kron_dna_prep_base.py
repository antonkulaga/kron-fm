"""The shared preparation base: atomic writes, digests, the layout's locate and split, name matching that refuses."""
from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest

from kron_dna.prep import io, layout as L


def toy(tmp_path):
    g = tmp_path / "toy"
    g.mkdir()
    np.arange(3000, dtype=np.uint8).tofile(g / "bases.u8")
    pl.DataFrame(dict(chrom=["chr1", "chr1", "chr2"], start=[0, 1600, 0], end=[1500, 2600, 500],
                      offset=[0, 1500, 2500])).write_parquet(g / "contigs.parquet")
    return L.Layout.open("toy", g)


def test_atomic_writes_whole_or_nothing(tmp_path):
    p = tmp_path / "a" / "f.json"
    io.write_json(p, {"x": 1})
    assert p.read_text().strip().startswith("{") and io.digest(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError):
        with io.atomic(tmp_path / "b.bin") as t:
            t.write_bytes(b"half")
            raise RuntimeError
    assert not (tmp_path / "b.bin").exists() and not list(tmp_path.glob("b.bin.part*"))


def test_layout_locates_and_splits(tmp_path):
    lay = toy(tmp_path)
    assert lay.total == 3000 and lay.rows()[1] == ("chr1", 1600, 2600, 1500)
    g = lay.locate(np.array(["chr1", "chr1", "chr1", "chr2", "chrX"]), np.array([10, 1550, 1700, 499, 5]))
    assert g.tolist() == [10, -1, 1600, 2999, -1]
    assert lay.split() is None
    (lay.path / "ntv3").mkdir()
    np.ones(3000, dtype=np.uint8).tofile(lay.path / "ntv3" / "split.u8")
    assert lay.split().sum() == 3000
    np.ones(10, dtype=np.uint8).tofile(tmp_path / "short.u8")
    with pytest.raises(ValueError):
        lay.split(tmp_path / "short.u8")


def test_names_match_or_refuse(tmp_path):
    lay = toy(tmp_path)
    assert L.match_names(["1", "2", "1"], lay) == {"1": "chr1", "2": "chr2"}
    assert L.match_names(["chr1", "MT"], lay, min_share=0.5) == {"chr1": "chr1"}
    with pytest.raises(ValueError, match="match the layout"):
        L.match_names(["NC_000001", "NC_000002"], lay)


def test_numpy_targets_equal_the_heads_torch_targets():
    """One target transform: the numpy twin (stores) and xheads.targets_of (training) agree on every byte value."""
    import torch
    from kron_dna import xheads as X
    from kron_dna.prep import transforms as T
    u8 = np.tile(np.arange(256, dtype=np.uint8)[:, None], (1, 4))
    scales, nz, rna = np.array([2.0, 4.0, 6.0, 9.0], np.float32), np.array([0.3, 1.0, 2.5, 0.7], np.float32), \
        np.array([False, True, False, True])
    ref = X.targets_of(torch.from_numpy(u8), torch.from_numpy(scales), torch.from_numpy(nz), torch.from_numpy(rna))
    got = T.targets_np(T.decode_u8(u8, scales) / nz, rna)
    np.testing.assert_allclose(got, ref.numpy(), rtol=2e-5, atol=1e-5)
    back = T.encode_u8(T.decode_u8(u8[:, 0], scales[0]), float(scales[0]))   # the encoding inverts its decoding
    assert np.abs(back.astype(int) - u8[:, 0]).max() <= 1


def test_one_intron_rule_with_a_minimum_length(tmp_path):
    """The shared reader and intron rule: GTF and GFF3 give the same table; gaps shorter than min_len are not introns."""
    from kron_dna.prep import features as F
    gtf = tmp_path / "a.gtf"
    gtf.write_text("".join(f"c1\tX\texon\t{a}\t{b}\t.\t+\t.\ttranscript_id \"t\";\n"
                           for a, b in ((1, 100), (103, 200), (301, 400))))
    gff = tmp_path / "a.gff3"
    gff.write_text("##gff-version 3\n" + "".join(f"c1\tX\texon\t{a}\t{b}\t.\t+\t.\tParent=t\n"
                                                for a, b in ((1, 100), (103, 200), (301, 400))))
    a, b = F.read_features(gtf), F.read_features(gff)
    assert a.drop("tid").equals(b.drop("tid"))
    assert sorted(F.introns(a, 1).select("s", "e").rows()) == [(100, 102), (200, 300)]
    assert F.introns(a, 4).select("s", "e").rows() == [(200, 300)]          # the 2-base gap is not a splice intron
