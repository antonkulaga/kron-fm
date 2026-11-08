"""kron-dna: a KronLSTM foundation model for coding DNA.

* :mod:`kron_dna.codons`: the 69-token codon vocabulary, the genetic code and coding-sequence checks.
* :mod:`kron_dna.data`: FASTA -> filtered, tokenized, packed rows; masking.
* :mod:`kron_dna.losses`: masked-LM, synonymous-constrained and hierarchical codon losses.
* :mod:`kron_dna.model`: the bidirectional KronLSTM + attention masked-language model.
* :mod:`kron_dna.cli`: the ``kron-dna`` command (prepare, train, info).
"""
