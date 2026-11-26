"""Download the ENCODE files a selection lists, each checked against ENCODE's own size and md5.

    uv run python scripts/kron_dna/encode_fetch.py SELECT.tsv OUT_DIR

``SELECT.tsv`` has at least the columns ``experiment``, ``file`` and ``url``. Each file goes to
``OUT_DIR/files/<file><the url's extension>.part`` with resume, is checked (size, md5 from ENCODE's file record), then renamed, so a
rerun skips what is complete. ``OUT_DIR/provenance.jsonl`` gets one line per verified file: accessions, assembly,
output type, replicates, lab, award, release date, size, md5, seconds and bytes downloaded in this run.
``OUT_DIR/status.tsv`` logs every attempt. The proxy is read from the environment by curl.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

API = "https://www.encodeproject.org"


def record(path: str) -> dict:
    out = subprocess.run(["curl", "-sS", "--fail", "--retry", "5", "--retry-all-errors", "--max-time", "120",
                          "-H", "Accept: application/json", f"{API}{path}?frame=object"],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main(select: Path, out: Path) -> int:
    files = out / "files"
    files.mkdir(parents=True, exist_ok=True)
    done = {}
    prov = out / "provenance.jsonl"
    if prov.exists():
        done = {r["file"]: r for r in map(json.loads, prov.read_text().splitlines())}
    failed = 0
    with open(select) as f, open(out / "status.tsv", "a") as status:
        for row in csv.DictReader(f, delimiter="\t"):
            ext = Path(row["url"]).suffix or ".bigWig"                 # .bigWig for tracks, .hic for contact maps
            acc, final = row["file"], files / f"{row['file']}{ext}"
            if acc in done and final.exists() and final.stat().st_size == done[acc]["size"]:
                continue
            meta = record(f"/files/{acc}/")
            exp = record(meta["dataset"])
            part = final.with_name(final.name + ".part")
            had = part.stat().st_size if part.exists() else 0
            t0, ok, why = time.time(), False, ""
            for attempt in range(5):
                subprocess.run(["curl", "-sSL", "--fail", "-C", "-", "--retry", "3", "--connect-timeout", "30",
                                "--speed-limit", "1024", "--speed-time", "120",
                                "-o", str(part), row["url"]])
                size = part.stat().st_size if part.exists() else 0
                if size == meta["file_size"]:
                    ok = md5(part) == meta["md5sum"]
                    why = "" if ok else "md5 mismatch"
                    if not ok:
                        part.unlink()
                        had = 0
                    else:
                        break
                elif size > meta["file_size"]:
                    part.unlink()
                    had, why = 0, "longer than ENCODE's size"
                else:
                    why = f"short: {size} of {meta['file_size']}"
                time.sleep(20 * (attempt + 1))
            seconds = time.time() - t0
            stamp = time.strftime("%Y-%m-%d %H:%M:%S%z")
            if not ok:
                failed += 1
                status.write(f"{stamp}\t{acc}\tFAILED\t{why}\n")
                status.flush()
                continue
            part.rename(final)
            line = {"file": acc, "experiment": row["experiment"], "assay": exp.get("assay_title"),
                    "biosample": exp.get("biosample_summary"), "assembly": meta.get("assembly"),
                    "output_type": meta.get("output_type"), "file_format": meta.get("file_format"),
                    "biological_replicates": meta.get("biological_replicates"), "lab": exp.get("lab"),
                    "award": exp.get("award"), "date_released": exp.get("date_released"),
                    "file_status": meta.get("status"), "size": meta["file_size"], "md5": meta["md5sum"],
                    "url": row["url"], "seconds": round(seconds, 1), "bytes_this_run": meta["file_size"] - had,
                    "fetched": stamp}
            with open(prov, "a") as p:
                p.write(json.dumps(line) + "\n")
            status.write(f"{stamp}\t{acc}\tok\t{meta['file_size']}\t{seconds:.0f}s\n")
            status.flush()
            print(f"{acc} ok {meta['file_size']} bytes in {seconds:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
