#!/usr/bin/env python3
"""Validate the pinned SVAMP challenge source and convert it to project JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


SVAMP_COMMIT = "78e727689e1c1bebfc4be39c446898e8e10b0518"
SVAMP_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
LICENSE_SHA256 = "07ce3a34de7a865345634f9278bc26ddcacc2d0defd348c28e3f4603448b9ba4"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/svamp-source/SVAMP.json"))
    parser.add_argument("--license", type=Path, default=Path("data/svamp-source/LICENSE"))
    parser.add_argument("--output", type=Path, default=Path("data/svamp/test.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/svamp/manifest.json"))
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    license_bytes = args.license.read_bytes()
    source_hash, license_hash = sha256(source_bytes), sha256(license_bytes)
    if source_hash != SVAMP_SHA256:
        raise SystemExit(f"Unexpected SVAMP.json SHA-256: {source_hash}")
    if license_hash != LICENSE_SHA256:
        raise SystemExit(f"Unexpected SVAMP LICENSE SHA-256: {license_hash}")

    raw_rows = json.loads(source_bytes)
    if not isinstance(raw_rows, list) or len(raw_rows) != 1000:
        raise SystemExit("Pinned SVAMP challenge source must contain exactly 1,000 rows.")
    required = {"ID", "Body", "Question", "Equation", "Answer", "Type"}
    ids: set[str] = set()
    questions: set[str] = set()
    output_rows = []
    for item in raw_rows:
        if not required.issubset(item):
            raise SystemExit(f"SVAMP row has missing fields: {item.get('ID')!r}")
        source_id = str(item["ID"])
        if source_id in ids:
            raise SystemExit(f"Duplicate SVAMP source ID: {source_id}")
        ids.add(source_id)
        prompt_text = f"{str(item['Body']).strip()}\n{str(item['Question']).strip()}"
        normalized_question = " ".join(prompt_text.split())
        if normalized_question in questions:
            raise SystemExit(f"Duplicate normalized SVAMP question: {source_id}")
        questions.add(normalized_question)
        answer = Decimal(str(item["Answer"]))
        if not answer.is_finite() or answer != answer.to_integral_value():
            raise SystemExit(f"Non-integer gold answer is unsupported by the verifier: {source_id}={item['Answer']!r}")
        output_rows.append(
            {
                "answer": str(int(answer)),
                "meta": {
                    "dataset": "SVAMP",
                    "dataset_revision": SVAMP_COMMIT,
                    "source_id": source_id,
                    "source_sha256": source_hash,
                    "type": str(item["Type"]),
                },
                "prompt": [
                    {
                        "content": (
                            f"{prompt_text}\n\nSolve the problem and show concise work. "
                            "Put only the final numeric answer, without units or commas, "
                            "inside <answer>...</answer>."
                        ),
                        "role": "user",
                    }
                ],
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output_rows)
    args.output.write_text(content, encoding="utf-8")
    manifest = {
        "dataset": "arkilpatel/SVAMP",
        "repository_commit": SVAMP_COMMIT,
        "source_file": args.source.as_posix(),
        "source_sha256": source_hash,
        "license_file": args.license.as_posix(),
        "license_sha256": license_hash,
        "rows": len(output_rows),
        "unique_source_ids": len(ids),
        "unique_normalized_questions": len(questions),
        "integer_gold_answers": len(output_rows),
        "output_file": args.output.as_posix(),
        "output_sha256": sha256(content.encode("utf-8")),
        "evaluation_role": "external_test_only",
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
