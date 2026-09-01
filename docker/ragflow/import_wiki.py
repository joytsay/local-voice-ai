"""Upload the local Markdown wiki to a RAGFlow dataset and start parsing it."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import httpx


def _checked_json(response: httpx.Response) -> Any:
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(payload.get("message", payload))
    return payload.get("data")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload knowledge/**/*.md and start RAGFlow parsing."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("RAGFLOW_BASE_URL", "http://ragflow-cpu:9380"),
    )
    parser.add_argument("--api-key", default=os.getenv("RAGFLOW_API_KEY", ""))
    parser.add_argument("--dataset-id", default=os.getenv("RAGFLOW_DATASET_IDS", ""))
    parser.add_argument("--wiki", type=Path, default=Path("/app/knowledge"))
    args = parser.parse_args()

    dataset_ids = [
        value.strip() for value in args.dataset_id.split(",") if value.strip()
    ]
    if not args.api_key:
        parser.error("provide --api-key or RAGFLOW_API_KEY")
    if len(dataset_ids) != 1:
        parser.error("the importer requires exactly one --dataset-id")
    pages = sorted(args.wiki.rglob("*.md"))
    if not pages:
        parser.error(f"no Markdown pages found under {args.wiki}")

    dataset_id = dataset_ids[0]
    headers = {"Authorization": f"Bearer {args.api_key}"}
    files = [
        (
            "file",
            (
                str(path.relative_to(args.wiki)).replace("/", "--"),
                path.read_bytes(),
                "text/markdown",
            ),
        )
        for path in pages
    ]
    with httpx.Client(base_url=args.base_url.rstrip("/"), timeout=300) as client:
        documents = _checked_json(
            client.post(
                f"/api/v1/datasets/{dataset_id}/documents",
                headers=headers,
                files=files,
            )
        )
        document_ids = [document["id"] for document in documents]
        _checked_json(
            client.post(
                f"/api/v1/datasets/{dataset_id}/chunks",
                headers=headers,
                json={"document_ids": document_ids},
            )
        )

    print(f"Uploaded and queued {len(document_ids)} wiki pages in {dataset_id}.")


if __name__ == "__main__":
    main()
