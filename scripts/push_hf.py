"""Upload only a validated dataset package to an existing Hugging Face repository."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from pde_rsi.hf_dataset import validate_package
from pde_rsi.storage import PROJECT_ROOT, project_path
from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="JeremyLu1017/PDE-RSI")
    parser.add_argument("--package", type=Path, default=PROJECT_ROOT/"datasets/huggingface/pde-problems")
    args = parser.parse_args()
    package = project_path(args.package)
    validation = validate_package(package)
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face login. Source scripts/env.sh, run hf auth login, then retry.")
    api = HfApi(token=token)
    account = api.whoami()["name"]
    info = api.repo_info(args.repo_id, repo_type="dataset")
    # Keep existing repository visibility and unrelated files; guard against a concurrent commit.
    manifest = json.loads((package/"manifest.json").read_text())
    allowed = sorted(set(manifest["files"]) | {"manifest.json"})
    operations = []
    for relative in allowed:
        path = project_path(package/relative)
        if not path.is_relative_to(package):
            raise ValueError("Upload source escaped the validated package")
        operations.append(CommitOperationAdd(path_in_repo=relative, path_or_fileobj=path))
    preflight = {"repository": args.repo_id, "account": account, "private": info.private,
                 "parent_commit": info.sha, "package": str(package), "files": allowed,
                 "validation": validation, "upload_authorization": "user requested this repository"}
    (PROJECT_ROOT/"runs/huggingface-upload-preflight.json").write_text(json.dumps(preflight, indent=2)+"\n")
    print(f"Uploading {len(operations)} reviewed files to {args.repo_id}; private={info.private}", flush=True)
    commit = api.create_commit(
        repo_id=args.repo_id, repo_type="dataset", operations=operations,
        parent_commit=info.sha, num_threads=2,
        commit_message="Add public PDE problem dataset and numerical solver bundle",
        commit_description=("48 manufactured PDE/FPDE cases across six typed configurations, "
                            "with train/validation splits, source provenance, and portable classical solvers. "
                            "Reserved final cases, references, environments, and RSI artifacts are excluded."))
    report = {"repository": args.repo_id, "url": f"https://huggingface.co/datasets/{args.repo_id}",
              "commit": commit.oid, "commit_url": commit.commit_url, "published": True,
              "private": info.private, "files_uploaded": len(allowed), "verified_files": 0,
              "public_cases": 48, "configs": 6, "reserved_final_cases_uploaded": False,
              "time_utc": datetime.now(timezone.utc).isoformat()}
    report_path = PROJECT_ROOT/"runs/huggingface-upload.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    remote = api.repo_info(args.repo_id, repo_type="dataset", revision=commit.oid, files_metadata=True)
    siblings = {item.rfilename: item for item in remote.siblings}
    for relative in allowed:
        if relative not in siblings:
            raise RuntimeError(f"Committed dataset file is missing: {relative}")
        local = (package/relative).read_bytes()
        sibling = siblings[relative]
        lfs = getattr(sibling, "lfs", None)
        if lfs is not None:
            remote_hash = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            matched = remote_hash == hashlib.sha256(local).hexdigest()
        elif getattr(sibling, "blob_id", None):
            expected = hashlib.sha1(f"blob {len(local)}\0".encode()+local).hexdigest()
            matched = expected == sibling.blob_id
        else:
            downloaded = hf_hub_download(args.repo_id, relative, repo_type="dataset", revision=commit.oid,
                                        token=token, cache_dir=PROJECT_ROOT/".runtime/cache/huggingface/upload-check")
            matched = Path(downloaded).read_bytes() == local
        if not matched:
            raise RuntimeError(f"Committed file does not match the reviewed package: {relative}")
        report["verified_files"] += 1
    report["remote_verification"] = "all package files matched"
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
