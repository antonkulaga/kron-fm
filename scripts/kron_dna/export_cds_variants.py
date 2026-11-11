"""The ClinVar variants kron-dna clinvar scores, each with its transcript's CDS, codon index and codons."""
import polars as pl
from kron_dna import variants as V
from kron_dna.cli import SOURCES, sequence_files
from kron_core.paths import input_dir, interim_dir
table = pl.read_parquet(interim_dir("cds", "clinvar") / "variants.parquet")
table = V.sample_per_class(table.filter(pl.col("consequence").is_in(["synonymous", "missense"])), 5000, 0)
tx = V.read_transcripts(sequence_files(SOURCES["human-grch38"].folder(input_dir()))[0])
cds = {t["name"]: t["seq"] for t in tx.values()}
out = table.select("id", "gene", "consequence", "label", "in_training", "transcript", "codon_index", "ref_codon",
                   "alt_codon").with_columns(pl.col("transcript").replace_strict(cds).alias("cds"))
out.write_parquet(interim_dir("cds", "clinvar") / "cds_variants.parquet")
print(out.height, "variants written")
