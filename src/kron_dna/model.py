"""kron-dna: a bidirectional KronLSTM masked-language model over codons.

A stack of :class:`kron_core.bidirectional.BiKronLSTMBlock` (a forward and a backward KronLSTM scan per
block), with a document-masked attention block after every ``attention_every``-th of them, over packed
rows of codon tokens. The default is 68.6M parameters (d_model 512, 12 recurrent blocks plus two
attention blocks; 51.9M with ``tie_directions``), in the 50-150M range the research report recommends
for a few A100-weeks. ``uv run kron-dna info`` prints the count of any configuration.

Design choices and where they come from (see ``reports/Small DNA foundation models 2026.md``):

* codon tokens, 1,024 per row, with packing: CodonMamba, SynCodonLM; packing is new here;
* two scans rather than one: the delta rule's erase term makes its state order-dependent;
* a few attention blocks: Gated DeltaNet's needle retrieval collapses past 4-8k tokens, and every
  scaled DNA hybrid keeps some attention (HybriDNA 7:1, StripedHyena 2, HAD);
* an optional species embedding added to every token, the token-type-ID design of SynCodonLM;
* no reverse-complement handling: the reverse complement of a coding strand is not itself coding.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from kron_core.bidirectional import BiKronLSTMBlock, SegmentAttentionBlock
from kron_core.segments import cu_seqlens_from_doc_ids, same_segment_mask
from kron_dna.codons import PAD_ID, VOCAB_SIZE
from kron_dna.data import doc_ids_from_tokens


@dataclass
class KronDNAConfig:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 12
    ff_units: int = 1536
    dropout: float = 0.1
    #: An attention block after every this many KronLSTM blocks; 0 for none.
    attention_every: int = 6
    #: One shared mixer for both scan directions (Caduceus's choice) instead of one per direction.
    tie_directions: bool = False
    #: Species (or species-cluster) ids 1..n_species get an embedding; 0 means unknown. 0 disables it.
    n_species: int = 0
    tie_embeddings: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class KronDNA(nn.Module):
    """``[B, T]`` codon tokens -> ``[B, T, vocab]`` logits (and the final hidden states)."""

    def __init__(self, config: KronDNAConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.embed = nn.Embedding(c.vocab_size, c.d_model, padding_idx=PAD_ID)
        self.species = nn.Embedding(c.n_species + 1, c.d_model, padding_idx=0) if c.n_species else None
        layers: list[nn.Module] = []
        for i in range(c.n_layers):
            layers.append(BiKronLSTMBlock(c.d_model, c.n_heads, c.ff_units, c.dropout, i, c.tie_directions))
            if c.attention_every and (i + 1) % c.attention_every == 0:
                layers.append(SegmentAttentionBlock(c.d_model, c.n_heads, c.ff_units, c.dropout))
        self.layers = nn.ModuleList(layers)
        self.final_norm = nn.RMSNorm(c.d_model, eps=1e-5)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self._init_weights()
        if c.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    @torch.no_grad()
    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            # the vendored layer initialises its own decay and gates (reset_gate_parameters) after this
            if isinstance(module, nn.Linear) and ".forward_mixer." not in f".{name}." \
                    and ".backward_mixer." not in f".{name}.":
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.embed.weight[PAD_ID].zero_()
        if self.species is not None:
            nn.init.zeros_(self.species.weight)

    def forward(self, tokens: torch.Tensor, doc_ids: torch.Tensor | None = None,
                species: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens`` ``[B, T]``; ``doc_ids`` defaults to one document per ``<cls>`` with padding apart;
        ``species`` is ``[B, T]`` (or ``[B]``, broadcast over the row) when the model has the embedding."""
        if doc_ids is None:
            doc_ids = doc_ids_from_tokens(tokens)
        h = self.embed(tokens)
        if self.species is not None and species is not None:
            h = h + self.species(species if species.dim() == 2 else species.unsqueeze(1).expand_as(tokens))
        cu_seqlens = cu_seqlens_from_doc_ids(doc_ids)
        segment_mask = same_segment_mask(doc_ids)
        for layer in self.layers:
            h = layer(h, cu_seqlens) if isinstance(layer, BiKronLSTMBlock) else layer(h, segment_mask)
        h = self.final_norm(h)
        return self.lm_head(h), h

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
