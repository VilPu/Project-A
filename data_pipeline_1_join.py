import pandas as pd
from pathlib import Path

INPUT_DIR = Path("datasets")
OUTPUT_DIR = Path("target")
OUTPUT_DIR.mkdir(exist_ok=True)

KEYS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded"
    ]

def main():
    # load data
    cohort = pd.read_csv(INPUT_DIR / "microbiology_cultures_cohort.csv", low_memory=False)

    ward = pd.read_csv(INPUT_DIR / "microbiology_cultures_ward_info.csv", low_memory=False)

    # remove duplicates
    cohort = cohort.drop_duplicates()
    ward = ward.drop_duplicates()

    # filter positive cultures
    cohort = cohort[cohort["was_positive"] == 1].copy()

    # only Susceptible and Resistant
    cohort["susceptibility"] = (cohort["susceptibility"
                                       ].astype("string")
                                       .str.strip()
                                       .str.lower()
    )

    cohort = cohort[cohort["susceptibility"
                           ].isin(["susceptible", "resistant"])
                           ].copy()

    # target
    cohort["target_susceptible"] = (cohort["susceptibility"] == "susceptible"
                                    ).astype(int)

    # join ward data
    ward_features = [
        "hosp_ward_IP",
        "hosp_ward_OP",
        "hosp_ward_ER",
        "hosp_ward_ICU"
    ]

    ward = ward[KEYS + ward_features]

    if ward.duplicated(KEYS).any():
        raise ValueError("Ward data has conflicting or repeated culture keys")

    df = cohort.merge(
        ward,
        on=KEYS,
        how="left",
        validate="many_to_one"
    )

    # save output
    df.to_csv(OUTPUT_DIR / "ml_candidates.csv",index=False)

    print("Final shape:", df.shape)

    print("\nTarget distribution:")
    print(df["target_susceptible"].value_counts())

    print("\nMissing values:")
    print(df.isna().mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()