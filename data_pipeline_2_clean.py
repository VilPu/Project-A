import pandas as pd
from pathlib import Path

INPUT = Path("target/ml_candidates.csv")
OUTPUT = Path("target/antibiotic_clean.csv")

KEYS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "organism",
    "antibiotic"
]

def main():
    df = pd.read_csv(INPUT)

    print("Original rows:", len(df))

    # only Susceptible and Resistant
    df = df[df["susceptibility"].isin([
        "susceptible", "resistant"
    ])].copy()

    # remove conflicting labels for the same organism + antibiotic
    conflicts = df.groupby(KEYS)["target_susceptible"].transform("nunique")
    df = df[conflicts == 1].copy()

    print("Rows after removing conflicts:", len(df))

    # remove duplicates
    df = df.drop_duplicates(subset=KEYS)

    print("Unique observations:", len(df))

    # remove columns and keep organism and antibiotic in the dataset for identification.
    remove_columns = [
        "susceptibility",
        "was_positive"
    ]

    df = df.drop(columns=remove_columns)

    # save output
    df.to_csv(OUTPUT, index=False)

    print("\nFinal shape:", df.shape)
    print("\nTarget distribution:")
    print(df["target_susceptible"].value_counts())
    print("\nSaved to:", OUTPUT)


if __name__ == "__main__":
    main()