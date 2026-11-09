"""Coding-sequence datasets kron-dna can download, into ``data/input/cds/<name>``.

The three NCBI RefSeq genomes are small and quick, for smoke runs and for looking at real CDS; the
SynCodonLM corpus is the pretraining set the research report recommends (43.5M coding sequences from
35,618 species, clustered per species at 90% nucleotide identity), one 17.4 GB gzipped CSV. Its first
rows come from one part of the table, so a ``--rows`` subset is for development, not a balanced sample.

NCBI ``cds_from_genomic`` files hold every annotated CDS, partial ones included (``[partial=5']``), and
every isoform of a human gene; :func:`kron_dna.codons.check_cds` drops the partial ones, and the human
isoforms are redundant until clustered.
"""
from __future__ import annotations

from kron_core.download import RemoteFile, Source, hf_dataset_url

DOMAIN = "cds"
NCBI = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF"
NCBI_LICENSE = "NCBI RefSeq: public domain in the United States (NCBI data usage policy)"
NCBI_CITATION = "O'Leary et al. (2016) Reference sequence (RefSeq) database at NCBI. Nucleic Acids Res 44:D733."


def _ncbi(accession: str, assembly: str) -> RemoteFile:
    digits = accession.split("_")[1].split(".")[0]
    path = "/".join(digits[i:i + 3] for i in range(0, 9, 3))
    return RemoteFile(f"{NCBI}/{path}/{accession}_{assembly}/{accession}_{assembly}_cds_from_genomic.fna.gz")


SOURCES: dict[str, Source] = {
    s.name: s for s in (
        Source("ecoli-k12", DOMAIN, "Escherichia coli K-12 MG1655 RefSeq CDS (GCF_000005845.2), about 4,300 genes",
               (_ncbi("GCF_000005845.2", "ASM584v2"),), NCBI_LICENSE, NCBI_CITATION,
               "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/",
               {"format": "fasta", "alt_starts": True, "approx_mb": 1.5}),
        Source("yeast-s288c", DOMAIN, "Saccharomyces cerevisiae S288C RefSeq CDS (GCF_000146045.2), about 6,000 genes",
               (_ncbi("GCF_000146045.2", "R64"),), NCBI_LICENSE, NCBI_CITATION,
               "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/",
               {"format": "fasta", "alt_starts": False, "approx_mb": 3.0}),
        Source("human-grch38", DOMAIN, "Homo sapiens GRCh38.p14 RefSeq CDS (GCF_000001405.40), every isoform",
               (_ncbi("GCF_000001405.40", "GRCh38.p14"),), NCBI_LICENSE, NCBI_CITATION,
               "https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000001405.40/",
               {"format": "fasta", "alt_starts": False, "approx_mb": 28.6}),
        Source("syncodonlm", DOMAIN,
               "SynCodonLM-V2 pretraining corpus: 43.5M RefSeq CDS from 35,618 species, MMseqs2-clustered per species "
               "at 90% identity / 80% coverage, columns CDS, Species, Group, Set (train/test)",
               (RemoteFile(hf_dataset_url("jheuschkel/clustered-cds-dataset", "cds-dataset-new.csv.gz"),
                           size=17_393_708_749),),
               "Apache-2.0",
               "Heuschkel et al. (2026) Advancing codon language modeling with synonymous codon constrained masking. "
               "Nucleic Acids Res 54(5):gkag166.",
               "https://huggingface.co/datasets/jheuschkel/clustered-cds-dataset",
               {"format": "csv", "sequence_column": "CDS", "species_column": "Species", "group_column": "Group",
                "set_column": "Set", "alt_starts": True, "approx_mb": 17_394}),
    )
}
