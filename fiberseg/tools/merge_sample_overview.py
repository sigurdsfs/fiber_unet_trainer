# merge_sample_overview.py
"""Merge a per-image metrics CSV (see `evaluate_predictions.py`) with sample metadata from
an overview spreadsheet (e.g. Sample_Overview_v4.xlsx).

Join logic
----------
Each image filename is expected to start with one of the overview sheet's `Sample` values
(e.g. "A33"), optionally followed by a "Site N" token whose N corresponds to that sample's
`Location` value in the sheet. Because the same `Sample` id can appear at more than one
`Location` in this data (different sampling campaigns reusing the same sample label), and
because not every filename's "Site N" actually matches an existing Location for that sample,
a match is only made when it is unambiguous:

- exactly one sheet row has that Sample -> matched on Sample alone
- multiple sheet rows share that Sample -> matched only if the filename's "Site N" equals
  exactly one of those rows' Location values

Anything else (no Sample prefix found, or a Sample with multiple rows that can't be resolved
by Site/Location) is left with blank overview columns and flagged via `match_status`, rather
than guessing, so ambiguous rows can be reviewed manually.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SITE_RE = re.compile(r"site\s*(\d+)", re.IGNORECASE)


def find_sample_id(filename: str, sample_ids: list[str]) -> str | None:
    """Return the longest Sample id that `filename` starts with, at a token boundary."""
    for sample_id in sample_ids:
        if not filename.startswith(sample_id):
            continue
        rest = filename[len(sample_id):]
        if not rest or not rest[0].isalnum():
            return sample_id
    return None


def find_site_number(filename: str) -> int | None:
    match = SITE_RE.search(filename)
    return int(match.group(1)) if match else None


def match_row(
    filename: str,
    overview: pd.DataFrame,
    sample_ids_by_length: list[str],
    sample_col: str,
    location_col: str,
) -> tuple[pd.Series | None, str | None, int | None, str]:
    sample_id = find_sample_id(filename, sample_ids_by_length)
    site_number = find_site_number(filename)

    if sample_id is None:
        return None, None, site_number, "no_sample_prefix_match"

    candidates = overview[overview[sample_col] == sample_id]

    if len(candidates) == 1:
        return candidates.iloc[0], sample_id, site_number, "matched_unique_sample"

    if site_number is None:
        return None, sample_id, site_number, "ambiguous_no_site_in_filename"

    sub = candidates[candidates[location_col] == site_number]
    if len(sub) == 1:
        return sub.iloc[0], sample_id, site_number, "matched_sample_and_site"
    if len(sub) == 0:
        return None, sample_id, site_number, "ambiguous_site_not_found"
    return None, sample_id, site_number, "ambiguous_multiple_site_matches"


def merge(
    metrics: pd.DataFrame,
    overview: pd.DataFrame,
    sample_col: str = "Sample",
    location_col: str = "Location",
    image_col: str = "image",
) -> pd.DataFrame:
    sample_ids_by_length = sorted(
        overview[sample_col].dropna().astype(str).unique(), key=len, reverse=True
    )

    matched_rows = []
    matched_samples = []
    matched_sites = []
    statuses = []

    for filename in metrics[image_col]:
        row, sample_id, site_number, status = match_row(
            filename, overview, sample_ids_by_length, sample_col, location_col
        )
        # dicts (not raw Series/None) so pd.DataFrame always reshapes into proper
        # columns regardless of how many unmatched (None) rows come first.
        matched_rows.append(row.to_dict() if row is not None else {})
        matched_samples.append(sample_id)
        matched_sites.append(site_number)
        statuses.append(status)

    overview_matches = pd.DataFrame(matched_rows, columns=overview.columns).reset_index(drop=True)

    result = metrics.reset_index(drop=True).copy()
    result["match_status"] = statuses
    result["matched_sample_id"] = matched_samples
    result["parsed_site_number"] = matched_sites
    result = pd.concat([result, overview_matches], axis=1)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Merge a per-image metrics CSV with sample metadata from an overview "
        "spreadsheet, matched via filename Sample/Site prefixes."
    )
    parser.add_argument("--metrics-csv", required=True, help="CSV from evaluate_predictions.py.")
    parser.add_argument("--overview-xlsx", required=True, help="Path to the overview spreadsheet.")
    parser.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet).")
    parser.add_argument("--sample-col", default="Sample")
    parser.add_argument("--location-col", default="Location")
    parser.add_argument("--image-col", default="image", help="Filename column in --metrics-csv.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: <metrics-csv-dir>/<metrics-csv-stem>_with_sample_info.csv).",
    )
    args = parser.parse_args()

    metrics_path = Path(args.metrics_csv)
    metrics = pd.read_csv(metrics_path)
    overview = pd.read_excel(args.overview_xlsx, sheet_name=args.sheet)

    result = merge(
        metrics,
        overview,
        sample_col=args.sample_col,
        location_col=args.location_col,
        image_col=args.image_col,
    )

    out_path = (
        Path(args.out)
        if args.out
        else metrics_path.with_name(f"{metrics_path.stem}_with_sample_info.csv")
    )
    result.to_csv(out_path, index=False)

    print(f"Wrote {len(result)} rows to {out_path}")
    print("Match status breakdown:")
    for status, count in result["match_status"].value_counts().items():
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
