#!/usr/bin/env python3
"""Convert GeoVision extension directory CSV into compact phonebook format.

Input columns expected:
公司/據點,頁籤/地點,部門/團隊,分機/電話,姓名,英文名,職稱,公司信箱

Output columns:
部門/團隊,中文名,英文名,分機
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


DEFAULT_INPUT = Path("geovision_extension_directory_clean.csv")
DEFAULT_OUTPUT = Path("phonebookall.csv")


def normalize_name(value: str) -> str:
    return " ".join(value.strip().split())


def parse_rows(input_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return rows

        for row in reader:
            if len(row) < 6:
                continue
            ext = row[3].strip()
            chinese_name = normalize_name(row[4])
            english_name = normalize_name(row[5])

            if not chinese_name or not ext:
                continue

            department = normalize_name(row[2])
            rows.append([department, chinese_name, english_name, ext])
    return rows


def write_rows(output_path: Path, rows: list[list[str]]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def main(argv: list[str]) -> int:
    input_path = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT
    output_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUTPUT

    if not input_path.exists():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 1

    rows = parse_rows(input_path)
    write_rows(output_path, rows)
    print(f"wrote {len(rows)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
