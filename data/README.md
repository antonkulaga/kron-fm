# Data

Everything kron-fm reads and writes lives under this folder (or under `$KRON_FM_DATA`, which moves the
whole tree). Only this README and the three folders' `.gitignore` files are in git.

| Folder | What goes in it | Who writes it |
|---|---|---|
| `input/<domain>/<source>/` | Datasets exactly as published, never edited. Each folder has a `SOURCE.json` with the URL, size, SHA-256, licence and citation of every file. | `uv run kron-data download <source>` |
| `interim/<domain>/<source>/` | Data derived from an input and rebuildable from it: packed token rows, splits. | `uv run kron-dna prepare <source>` |
| `output/runs/<model>/<run>/` | Training runs: `checkpoint.pt`, `config.json`, `metrics.csv`, and `onnx/` after export. | `uv run kron-dna train`, `uv run kron-dna export-onnx` |

`uv run kron-data list` shows every dataset kron-fm knows, and `uv run kron-data status` shows what is on
disk. Domains so far: `cds` (coding sequences for kron-dna) and `timeseries` (krontime; MetaboNet is a link to
the glucose repository's download).
