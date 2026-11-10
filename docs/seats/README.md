# Seats

This repository runs two seats of the seats kit (agent-seat-blueprints), each in a clone of its own on the cluster,
both of the kit's role `trainer`:

| Seat | Priming | Trains | Its jobs |
|---|---|---|---|
| `ppg-trainer` | `PPG_TRAINER.md` | `kron_ppg`, the streaming PPG encoder and its aging clock | `kron-ppg-*` |
| `dna-trainer` | `DNA_TRAINER.md` | `kron_dna`, the codon masked-language model | `kron-dna-*` |

The seat table is `.blueprints.toml` at the root. Each seat's working state is its `<SEAT>_NOTES.md` beside its
priming, written by the seat itself.

The primings write `<brief>`, `<status file>` and `<share>` where a fact belongs to the cluster or to the
maintainers' plan: the brief's path, the status file's, and the seat's cores, GPUs and memory per job, which the
organisation's rigger keeps. Each seat writes them into its notes at its first seating, never a machine's fact.

[encrypted, signed by agents-kit/builder@longevity-crew.zulipchat.com, checked against secretary's contacts]
kron-fm seat files, part 2 of 2: the two primings.
