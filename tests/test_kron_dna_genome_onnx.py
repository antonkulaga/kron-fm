"""kron_dna.genome_onnx: KronGenome exported to ONNX through the plain KronLSTM scan (CPU, a tiny config)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from kron_dna import genome as G
from kron_dna import genome_onnx as GO
from test_kron_dna_multigenome import genomes  # noqa: F401  (a fixture)
from test_kron_dna_xheads_cli import make_store

pytest.importorskip("onnxruntime")


def tiny_checkpoint(tmp_path, hgt, bin_dnase: int = 1) -> tuple:
    """A tiny KronGenome (real KronLSTM layers) with an NTv3 head and KronLSTM further heads, saved as a checkpoint
    whose run record points at stores, as genome-train writes it."""
    store = make_store(tmp_path, hgt, n=3)
    total = int(np.fromfile(hgt / "bases.u8", dtype=np.uint8).size)
    lab = tmp_path / "splice"
    lab.mkdir()
    np.zeros(total, dtype=np.uint8).tofile(lab / "splice.u8")
    bits = {"donor_plus": 0, "acceptor_plus": 1, "donor_minus": 2, "acceptor_minus": 3}
    (lab / "splice.json").write_text(json.dumps(dict(genome="hgt", total_bases=total, bits=bits, definition="x",
                                                     splice_sha256="x", check={k: dict(sites=1) for k in bits})))
    nd = tmp_path / "ntv3"
    nd.mkdir()
    ids = [f"T{k}" for k in range(4)] + ["R_P", "R_M"]
    (nd / "dense.json").write_text(json.dumps(dict(ids=ids, nonzero_mean=[0.5, 1.0, 2.0, 0.7, 0.3, 0.4])))
    (nd / "tracks.json").write_text(json.dumps(dict(tracks=[dict(file_id=i, assay="RNA-seq" if i.startswith("R") else "ATAC-seq")
                                                             for i in ids])))
    cfg = G.KronGenomeConfig(d_model=32, n_heads=2, n_layers=2, ff_units=64, stem_channels=8, n_down=4, arch="v2",
                             n_ntv3=6, n_contexts=8, xhead_kind="kron", xhead_bin=4, xhead_width=64, xhead_layers=1,
                             xheads=[["dnase", 3] + ([bin_dnase] if bin_dnase > 1 else []), ["splice", 4]])
    torch.manual_seed(0)
    model = G.KronGenome(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05 * torch.randn_like(p))
    st = XS = None
    from kron_dna import xheads as X
    st, XS = X.open_store(store), X.open_store(lab)
    record = dict(options=dict(ntv3_dir=str(nd), ntv3_extra=""),
                  xheads=dict(stores=dict(dnase=dict(path=str(store), **st.identity), splice=dict(path=str(lab), **XS.identity))))
    ck = tmp_path / "ck.pt"
    torch.save(dict(model=model.state_dict(), config=cfg.to_dict(), step=11, run_record=record,
                    ntv3=dict(ids=ids, loss="poisson", target="alphagenome")), ck)
    return ck, ids


@pytest.mark.parametrize("bin_dnase", [1, 16])
def test_export_matches_pytorch_and_names_every_output(genomes, tmp_path, bin_dnase):
    ck, ids = tiny_checkpoint(tmp_path, genomes["hgt"], bin_dnase)
    meta = GO.export_genome_onnx(ck, tmp_path / "onnx", window=256, check_kernels=False)
    names = [o["name"] for o in meta["outputs"]]
    assert names == ["base_logits", "ntv3", "x_dnase", "x_splice"]
    shapes = {o["name"]: o["shape"] for o in meta["outputs"]}
    assert shapes["ntv3"] == ["batch", 256, 6] and shapes["x_dnase"] == ["batch", 256 // bin_dnase, 3]
    assert shapes["x_splice"] == ["batch", 256, 4]
    for v in meta["verification"]:
        assert v["max_abs_diff"] <= 1e-4 * max(1.0, v["max_abs_value"]), v
    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp_path / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"bases": GO.example_bases(3, 256, seed=5).numpy()})            # a free batch size
    assert [o.shape[0] for o in out] == [3, 3, 3, 3] and (out[1] >= 0).all() and ((out[3] >= 0) & (out[3] <= 1)).all()


def test_the_plain_graph_reads_each_row_alone(genomes, tmp_path):
    """Rows are documents: a row's outputs do not depend on the other rows of the batch."""
    ck, _ = tiny_checkpoint(tmp_path, genomes["hgt"])
    model, state = G.load_checkpoint(ck)
    graph = GO.KronGenomeGraph(model, None, None, {}, window=256).eval()
    x = GO.example_bases(2, 256, seed=3)
    with torch.no_grad():
        both = graph(x)
        one = graph(x[1:])
    for a, b in zip(both, one):
        assert torch.allclose(a[1:], b, atol=1e-5)


def test_natural_units_equal_the_training_inverse():
    from kron_dna import xheads as X
    raw = torch.linspace(-5, 30, 50).reshape(5, 10, 1).expand(5, 10, 2)
    nz, rna = torch.tensor([0.4, 2.0]), torch.tensor([False, True])
    want = torch.expm1(X.signal_of(torch.nn.functional.softplus(raw), nz, rna))
    assert torch.allclose(GO.natural(raw, nz, rna), want, rtol=1e-4, atol=1e-5)


def test_the_graphs_forward_is_the_models_forward(genomes, tmp_path, device):
    """run_model (no segments, plain row starts) gives KronGenome.forward's outputs, on the model's own layers (on
    the CPU with the stand-in for the KronLSTM layer, which both paths call alike)."""
    if device.type != "cpu":
        pytest.skip("the CPU case is the stand-in one")
    ck, _ = tiny_checkpoint(tmp_path, genomes["hgt"])
    model, _ = G.load_checkpoint(ck)
    x = GO.example_bases(2, 256, seed=2)
    with torch.no_grad():
        want = model(x, None, torch.zeros(2, dtype=torch.long), None)[0]
        nt, xs = model.last_ntv3.clone(), {k: v.clone() for k, v in model.last_x.items()}
        got = GO.KronGenomeGraph.run_model(model, x)
    assert torch.allclose(got, want, atol=1e-6) and torch.allclose(model.last_ntv3, nt, atol=1e-6)
    assert all(torch.allclose(model.last_x[k], xs[k], atol=1e-6) for k in xs)
