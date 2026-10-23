import pandas as pd
from pathlib import Path

INPUT = Path("target/antibiotic_clean.csv")
OUTPUT = Path("results/dataset_profile.txt")
OUTPUT.parent.mkdir(exist_ok=True)

KEYS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded",
    "organism",
    "antibiotic"
]

CULTURE_KEYS = [
    "anon_id",
    "pat_enc_csn_id_coded",
    "order_proc_id_coded"
]


def main():
    df = pd.read_csv(INPUT)

    df["order_time_jittered_utc"] = pd.to_datetime(
        df["order_time_jittered_utc"], utc=True
    )

    profile = {
        "observations": len(df),
        "patients": df["anon_id"].nunique(),
        "culture_events": df[CULTURE_KEYS].drop_duplicates().shape[0],
        "organisms": df["organism"].nunique(),
        "antibiotics": df["antibiotic"].nunique(),
        "first_culture": df["order_time_jittered_utc"].min(),
        "last_culture": df["order_time_jittered_utc"].max(),
        "duplicate_keys": df.duplicated(KEYS).sum(),
        "susceptible": (df["target_susceptible"] == 1).sum(),
        "resistant": (df["target_susceptible"] == 0).sum(),
        "susceptible_rate": (df["target_susceptible"] == 1).mean(),
        "resistant_rate": (df["target_susceptible"] == 0).mean()
    }

    depth = df.groupby(CULTURE_KEYS + ["organism"])["antibiotic"].nunique()

    missing = pd.DataFrame({
        "count": df.isna().sum(),
        "percent": df.isna().mean() * 100
    })

    antibiotics = pd.crosstab(
        df["antibiotic"],
        df["target_susceptible"]
    ).reindex(columns=[0, 1], fill_value=0)

    antibiotics.columns = ["resistant", "susceptible"]
    antibiotics["total"] = antibiotics.sum(axis=1)
    antibiotics["resistant_percent"] = (
        antibiotics["resistant"] / antibiotics["total"] * 100
    )

    antibiotics = antibiotics.sort_values("total", ascending=False)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("DATASET PROFILE\n\n")

        for key, value in profile.items():
            f.write(f"{key}: {value}\n")

        f.write("\nCANDIDATE DEPTH PER CULTURE AND ORGANISM\n")
        f.write(depth.describe().to_string())

        f.write("\n\nMISSING VALUES\n")
        f.write(missing.to_string())

        f.write("\n\nANTIBIOTIC DISTRIBUTION\n")
        f.write(antibiotics.to_string())

        f.write("\n")

    print(f"Dataset profile saved to {OUTPUT}")


if __name__ == "__main__":
    main()