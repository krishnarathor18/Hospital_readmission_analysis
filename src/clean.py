
#Cleaning and feature engineering for the UCI Diabetes 130-US Hospitals dataset.
import pandas as pd
import numpy as np


ICD9_MAP_RANGES = [
    (390, 459, "Circulatory"),
    (460, 519, "Respiratory"),
    (520, 579, "Digestive"),
    (580, 629, "Genitourinary"),
    (140, 239, "Neoplasms"),
    (710, 739, "Musculoskeletal"),
    (800, 999, "Injury"),
    (240, 279, "Endocrine/Metabolic"),
    (290, 319, "Mental"),
    (320, 389, "Nervous System"),
]


def map_diagnosis(code):
    """Map an ICD-9 diagnosis code to a broad clinical category."""
    if pd.isna(code) or code == "?":
        return "Missing"
    code = str(code)
    if code.startswith("250"):
        return "Diabetes"
    if code.startswith(("V", "E")):
        return "Other/External"
    try:
        val = float(code)
    except ValueError:
        return "Other/External"
    for low, high, label in ICD9_MAP_RANGES:
        if low <= val <= high:
            return label
    return "Other"


def load_and_clean(path):
    df = pd.read_csv(path)

    # Replace '?' with NAN
    df = df.replace("?", np.nan)

    # Drop columns that are almost entirely missing 
    df = df.drop(columns=["weight", "payer_code", "encounter_id"])

    df["medical_specialty"] = df["medical_specialty"].fillna("Missing")

    # race: small amount missing so fill as "Unknown"
    df["race"] = df["race"].fillna("Unknown")

  
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Not Tested")
    df["A1Cresult"] = df["A1Cresult"].fillna("Not Tested")

    # Drop the small number of rows with unknown gender
    df = df[df["gender"] != "Unknown/Invalid"]

    # Diagnosis columns -> broad clinical category
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col + "_cat"] = df[col].apply(map_diagnosis)
    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

  
    expired_codes = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(expired_codes)]

    
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    def age_midpoint(bracket):
        low, high = bracket.strip("[)").split("-")
        return (int(low) + int(high)) / 2

    df["age_numeric"] = df["age"].apply(age_midpoint)

    return df


NUMERIC_FEATURES = [
    "age_numeric",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "diag_1_cat",
    "diag_2_cat",
    "diag_3_cat",
    "insulin",
    "change",
    "diabetesMed",
]

TARGET = "readmitted_30d"


if __name__ == "__main__":
    df = load_and_clean("/home/claude/readmission_project/data/diabetic_data.csv")
    print("Cleaned shape:", df.shape)
    print(df[TARGET].value_counts(normalize=True))
    print(df.isnull().sum()[df.isnull().sum() > 0])
