#!/usr/bin/env python3
"""List and (selectively) download Amazon's BodyM dataset from its public S3
bucket, with the standard library only -- no AWS CLI, no AWS account.

BodyM (CC BY-NC 4.0; https://registry.opendata.aws/bodym/): 2,505 subjects
with height, weight, gender and 14 body measurements (cm), plus frontal and
lateral BLACK-AND-WHITE SILHOUETTES (not RGB photos). Bucket: amazon-bodym,
region us-west-2.

The file layout is not documented where we read it, so this script works in
two steps instead of guessing:
  1. `--list`      prints what is in the bucket: top-level folders, file
                   counts per extension, sizes. Downloads nothing.
  2. `--download`  fetches only the files you select (default: small table/text
                   files <= --max-mb, extensions csv/tsv/json/txt/md), so the
                   silhouettes stay on S3 unless you ask for them with --ext.
                   Prints the plan (count, total size) before downloading.

Usage (self-test, no network):
    python data_prep/fetch_bodym.py --self-test

Usage:
    python data_prep/fetch_bodym.py --list
    python data_prep/fetch_bodym.py --download --dest ~/datasets/BodyM
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

BUCKET = "amazon-bodym"
REGION = "us-west-2"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DEFAULT_EXT = ("csv", "tsv", "json", "txt", "md")


def _http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def parse_listing(xml_bytes: bytes) -> tuple[list[tuple[str, int]], str | None]:
    """One ListObjectsV2 page -> ([(key, size)], next continuation token or None)."""
    root = ET.fromstring(xml_bytes)
    items = [(c.findtext(f"{NS}Key"), int(c.findtext(f"{NS}Size"))) for c in root.findall(f"{NS}Contents")]
    token = root.findtext(f"{NS}NextContinuationToken") if root.findtext(f"{NS}IsTruncated") == "true" else None
    return items, token


def list_all(http_get=_http_get, bucket: str = BUCKET, region: str = REGION) -> list[tuple[str, int]]:
    base = f"https://{bucket}.s3.{region}.amazonaws.com/"
    out, token = [], None
    while True:
        q = {"list-type": "2"}
        if token:
            q["continuation-token"] = token
        items, token = parse_listing(http_get(base + "?" + urllib.parse.urlencode(q)))
        out += [(k, s) for k, s in items if not k.endswith("/")]
        if not token:
            return out


def ext_of(key: str) -> str:
    return key.rsplit(".", 1)[-1].lower() if "." in key.rsplit("/", 1)[-1] else "(none)"


def summarize(files: list[tuple[str, int]]) -> list[str]:
    lines = [f"{len(files)} file(s), {sum(s for _, s in files) / 1e6:,.1f} MB in total", ""]
    by_dir = defaultdict(list)
    for k, s in files:
        by_dir[k.split("/")[0] if "/" in k else "(root)"].append((k, s))
    lines.append(f"{'top-level folder':34s}{'files':>8s}{'MB':>10s}  extensions")
    for d, items in sorted(by_dir.items()):
        exts = dict(Counter(ext_of(k) for k, _ in items))
        lines.append(f"{d:34s}{len(items):8d}{sum(s for _, s in items) / 1e6:10.1f}  {exts}")
    small = sorted((k, s) for k, s in files if ext_of(k) in DEFAULT_EXT)
    lines += ["", "table/text files (what --download fetches by default):"]
    lines += [f"  {s / 1e6:9.2f} MB  {k}" for k, s in small[:40]]
    if len(small) > 40:
        lines.append(f"  ... and {len(small) - 40} more")
    lines += ["", "first few keys of each folder:"]
    for d, items in sorted(by_dir.items()):
        lines += [f"  {k}" for k, _ in sorted(items)[:3]]
    return lines


def select(files, exts, max_mb: float, pattern: str | None = None):
    exts = {e.lower().lstrip(".") for e in exts}
    return [(k, s) for k, s in files
            if ext_of(k) in exts and s <= max_mb * 1e6 and (pattern is None or pattern in k)]


def download(chosen, dest: Path, http_get=_http_get, bucket: str = BUCKET, region: str = REGION) -> int:
    n = 0
    for k, s in chosen:
        target = dest / k
        if target.exists() and target.stat().st_size == s:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://{bucket}.s3.{region}.amazonaws.com/" + urllib.parse.quote(k)
        target.write_bytes(http_get(url))
        n += 1
        print(f"  downloaded {k} ({s / 1e6:.2f} MB)", flush=True)
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show what is in the bucket; download nothing")
    ap.add_argument("--download", action="store_true", help="download the selected files")
    ap.add_argument("--dest", default="~/datasets/BodyM")
    ap.add_argument("--ext", nargs="+", default=list(DEFAULT_EXT),
                    help="extensions to download (add png/jpg only if you really want the silhouettes)")
    ap.add_argument("--max-mb", type=float, default=200.0, help="skip single files larger than this")
    ap.add_argument("--pattern", default=None, help="only keys containing this text")
    args = ap.parse_args()
    if not (args.list or args.download):
        raise SystemExit("choose --list or --download")

    print(f"listing s3://{BUCKET} ({REGION}) ...", flush=True)
    files = list_all()
    if args.list:
        print("\n".join(summarize(files)))
    if args.download:
        chosen = select(files, args.ext, args.max_mb, args.pattern)
        total = sum(s for _, s in chosen) / 1e6
        print(f"\nplan: {len(chosen)} file(s), {total:,.1f} MB, extensions {args.ext}, into {args.dest}")
        if not chosen:
            raise SystemExit("nothing matches -- run --list and adjust --ext / --pattern")
        n = download(chosen, Path(args.dest).expanduser())
        print(f"done: {n} new file(s) (existing identical files skipped)")


def self_test() -> None:
    def page(items, token=None):
        body = "".join(f"<Contents><Key>{k}</Key><Size>{s}</Size></Contents>" for k, s in items)
        extra = f"<IsTruncated>true</IsTruncated><NextContinuationToken>{token}</NextContinuationToken>" if token \
            else "<IsTruncated>false</IsTruncated>"
        return f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{body}{extra}</ListBucketResult>'.encode()

    pages = {None: page([("README.md", 1200), ("train/", 0), ("train/subject_measurements.csv", 90000)], "t1"),
             "t1": page([("train/silhouettes/a_front.png", 4000), ("test_a/m.csv", 5000), ("big.csv", 500_000_000)])}
    calls = []

    def fake_get(url):
        calls.append(url)
        if "?list-type" in url:
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            return pages[q.get("continuation-token", [None])[0]]
        return b"x" * 7

    files = list_all(fake_get)
    assert [k for k, _ in files] == ["README.md", "train/subject_measurements.csv",
                                     "train/silhouettes/a_front.png", "test_a/m.csv", "big.csv"], files
    text = "\n".join(summarize(files))
    assert "5 file(s)" in text and "train" in text and "{'csv': 1, 'png': 1}" in text, text
    chosen = select(files, DEFAULT_EXT, max_mb=200)
    assert [k for k, _ in chosen] == ["README.md", "train/subject_measurements.csv", "test_a/m.csv"]
    assert [k for k, _ in select(files, ["png"], 200)] == ["train/silhouettes/a_front.png"]
    assert [k for k, _ in select(files, DEFAULT_EXT, 200, "test_a")] == ["test_a/m.csv"]

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        n = download([("test_a/m.csv", 7)], Path(d), fake_get)
        assert n == 1 and (Path(d) / "test_a" / "m.csv").read_bytes() == b"x" * 7
        assert download([("test_a/m.csv", 7)], Path(d), fake_get) == 0, "identical file must be skipped"
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
