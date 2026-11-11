"""Genomic positions (chrom, pos, ref, alt) of the ClinVar variants kron-dna clinvar scores, by id, for Evo 2."""
import polars as pl
from kron_dna import variants as V
from kron_core.paths import input_dir, interim_dir
table = pl.read_parquet(interim_dir("cds", "clinvar") / "variants.parquet")
table = V.sample_per_class(table.filter(pl.col("consequence").is_in(["synonymous", "missense"])), 5000, 0)
pos = V.read_clinvar(input_dir("variants", "clinvar") / "clinvar.vcf.gz").filter(pl.col("id").is_in(table["id"].implode()))
out = table.select("id", "consequence", "label").join(pos.select("id", "chrom", "pos", "ref", "alt"), on="id")
out.write_parquet(interim_dir("cds", "clinvar") / "positions.parquet")
print(out.height, "positions written")
