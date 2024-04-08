import argparse

import pandas as pd
from sklearn.metrics import mutual_info_score


def DataFile(path):
    full = pd.read_csv(path)
    if "exp" not in full.columns:
        raise ValueError('Data file must contain "exp" column')
    if "cluster" not in full.columns:
        raise ValueError('Data file must contain "cluster" column')

    has_unit = "unit" in full.columns
    has_id = "id" in full.columns
    if has_unit == has_id:
        raise ValueError('Data file must contain either "unit" or "id" column')
    elif has_id:
        full = full.rename(columns={"id": "unit"})
    return full[["exp", "unit", "cluster"]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            Calculate mutual information between two clusterings, as well as a p-value
            for that score. Also optionally output a CSV with each of the units and
            their values in each of the clusterings for other analysis."""
    )
    parser.add_argument("FILE", nargs=2, help="Path to data file", type=DataFile)
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output CSV file",
        nargs="?",
        type=argparse.FileType("w"),
        default=None,
    )
    parser.add_argument(
        "-N",
        help="Number of surrogates to generate for p-value calculation",
        type=int,
        default=2000,
    )
    args = parser.parse_args()
    A, B = args.FILE

    joined = A.merge(B, on=["exp", "unit"], suffixes=("_A", "_B"))
    if args.output:
        joined.to_csv(args.output, index=False)

    mi = mutual_info_score(joined.cluster_A, joined.cluster_B)
    mi_surrs = [
        mutual_info_score(
            joined.cluster_A.sample(frac=1), joined.cluster_B.sample(frac=1)
        )
        for _ in range(args.N)
    ]

    p = sum(mi_surr > mi for mi_surr in mi_surrs) / args.N
    if p == 0:
        print(f"MI: {mi:.3f}, surr MI range {min(mi_surrs):.3f} - {max(mi_surrs):.3f}.")
    else:
        print(f"MI: {mi:.3f}, {p = }")
