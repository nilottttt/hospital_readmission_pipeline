"""Generate a SMALL SYNTHETIC file with the same layout as the UCI dataset.

FOR OFFLINE TESTING ONLY - the real pipeline downloads the real UCI data.
It deliberately includes dirty rows ('?', Unknown/Invalid gender, duplicates,
expired patients, a malformed line) so every validation rule is exercised.

    python scripts/generate_sample_data.py --rows 5000 --out data/manual
"""
import argparse
import csv
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etl.columns import MEDICATION_SOURCE_COLUMNS, SOURCE_COLUMNS, VALID_AGE_BRACKETS  # noqa: E402

MAPPING_TEXT = """admission_type_id,description
1,Emergency
2,Urgent
3,Elective
4,Newborn
5,Not Available
6,NULL
7,Trauma Center
8,Not Mapped
,
discharge_disposition_id,description
1,Discharged to home
2,Discharged/transferred to another short term hospital
3,Discharged/transferred to SNF
4,Discharged/transferred to ICF
5,Discharged/transferred to another type of inpatient care institution
6,Discharged/transferred to home with home health service
7,Left AMA
8,Discharged/transferred to home under care of Home IV provider
9,Admitted as an inpatient to this hospital
10,Neonate discharged to another hospital for neonatal aftercare
11,Expired
12,Still patient or expected to return for outpatient services
13,Hospice / home
14,Hospice / medical facility
15,Discharged/transferred within this institution to Medicare approved swing bed
16,Discharged/transferred/referred another institution for outpatient services
17,Discharged/transferred/referred to this institution for outpatient services
18,NULL
19,"Expired at home. Medicaid only, hospice."
20,"Expired in a medical facility. Medicaid only, hospice."
21,"Expired, place unknown. Medicaid only, hospice."
22,Discharged/transferred to another rehab fac including rehab units of a hospital .
23,Discharged/transferred to a long term care hospital.
24,Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.
25,Not Mapped
26,Unknown/Invalid
30,Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere
27,Discharged/transferred to a federal health care facility.
28,Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital
29,Discharged/transferred to a Critical Access Hospital (CAH).
,
admission_source_id,description
1, Physician Referral
2,Clinic Referral
3,HMO Referral
4,Transfer from a hospital
5, Transfer from a Skilled Nursing Facility (SNF)
6, Transfer from another health care facility
7, Emergency Room
8, Court/Law Enforcement
9, Not Available
10, Transfer from critial access hospital
11,Normal Delivery
12, Premature Delivery
13, Sick Baby
14, Extramural Birth
15,Not Available
17,NULL
18, Transfer From Another Home Health Agency
19,Readmission to Same Home Health Agency
20, Not Mapped
21,Unknown/Invalid
22, Transfer from hospital inpt/same fac reslt in a sep claim
23, Born inside this hospital
24, Born outside this hospital
25, Transfer from Ambulatory Surgery Center
26,Transfer from Hospice
"""

SPECIALTIES = ["?"] * 10 + ["InternalMedicine"] * 3 + ["Emergency/Trauma"] * 2 + [
    "Family/GeneralPractice", "Cardiology", "Surgery-General", "Nephrology",
    "Orthopedics", "Orthopedics-Reconstructive", "Radiologist", "Pulmonology",
    "Psychiatry", "Urology", "ObstetricsandGynecology", "Gastroenterology",
    "Hematology/Oncology", "Pediatrics-Endocrinology", "Neurology"]
DIAGS = ["428", "414", "786", "410", "427", "486", "491", "715", "682", "434",
         "780", "996", "276", "250.8", "250.6", "250.02", "599", "584", "V57", "E888",
         "435", "8", "562", "577", "038", "185", "?"]


def weighted(options):
    vals, w = zip(*options)
    return random.choices(vals, weights=w)[0]


def make_row(enc_id, pat_id):
    age_idx = min(9, max(0, int(random.gauss(6.5, 1.6))))
    inpatient = weighted([(0, 66), (1, 19), (2, 7), (3, 4), (5, 4)])
    emergency = weighted([(0, 89), (1, 8), (2, 3)])
    los = min(14, max(1, int(random.expovariate(1 / 4.4)) + 1))
    nmeds = max(1, int(random.gauss(16, 8)))
    ndiag = weighted([(9, 49), (8, 10), (7, 10), (6, 10), (5, 11), (4, 5), (3, 3), (16, 2)])
    base = 0.07 + 0.03 * inpatient + 0.02 * (los >= 7) + 0.01 * (age_idx >= 7)
    r = random.random()
    readm = "<30" if r < min(base, 0.4) else (">30" if r < min(base, 0.4) + 0.35 else "NO")
    meds = {m: weighted([("No", 80), ("Steady", 15), ("Up", 3), ("Down", 2)])
            for m in MEDICATION_SOURCE_COLUMNS}
    meds["insulin"] = weighted([("No", 47), ("Steady", 30), ("Up", 11), ("Down", 12)])
    changed = any(v in ("Up", "Down") for v in meds.values())
    row = {
        "encounter_id": enc_id, "patient_nbr": pat_id,
        "race": weighted([("Caucasian", 75), ("AfricanAmerican", 19), ("?", 2),
                          ("Hispanic", 2), ("Other", 1.5), ("Asian", 0.5)]),
        "gender": weighted([("Female", 54), ("Male", 46), ("Unknown/Invalid", 0.2)]),
        "age": VALID_AGE_BRACKETS[age_idx],
        "weight": weighted([("?", 97), ("[75-100)", 2), ("[50-75)", 1)]),
        "admission_type_id": weighted([(1, 53), (2, 18), (3, 19), (6, 5), (5, 5), (8, 0.3), (7, 0.1)]),
        "discharge_disposition_id": weighted([(1, 59), (3, 14), (6, 13), (18, 4), (2, 2), (22, 2),
                                              (11, 1.6), (5, 1.2), (25, 1), (4, 0.8), (7, 0.6),
                                              (23, 0.4), (13, 0.4), (14, 0.4), (28, 0.1)]),
        "admission_source_id": weighted([(7, 56), (1, 29), (17, 7), (4, 3), (6, 2), (2, 1), (5, 1), (20, 0.2), (9, 0.1)]),
        "time_in_hospital": los,
        "payer_code": weighted([("?", 40), ("MC", 32), ("HM", 6), ("SP", 5), ("BC", 5), ("MD", 4), ("CP", 2), ("UN", 2), ("CM", 2)]),
        "medical_specialty": random.choice(SPECIALTIES),
        "num_lab_procedures": max(1, int(random.gauss(43, 20))),
        "num_procedures": weighted([(0, 46), (1, 20), (2, 12), (3, 9), (6, 5), (4, 4), (5, 3)]),
        "num_medications": nmeds,
        "number_outpatient": weighted([(0, 84), (1, 8), (2, 4), (3, 4)]),
        "number_emergency": emergency, "number_inpatient": inpatient,
        "diag_1": random.choice(DIAGS[:-1]), "diag_2": random.choice(DIAGS), "diag_3": random.choice(DIAGS),
        "number_diagnoses": ndiag,
        "max_glu_serum": weighted([("None", 95), ("Norm", 2.5), (">200", 1.5), (">300", 1)]),
        "A1Cresult": weighted([("None", 83), (">8", 8), ("Norm", 5), (">7", 4)]),
        **meds,
        "change": "Ch" if changed else "No",
        "diabetesMed": "Yes" if changed or any(v != "No" for v in meds.values()) else "No",
        "readmitted": readm,
    }
    return [row[c] for c in SOURCE_COLUMNS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=5000)
    ap.add_argument("--out", default="data/manual")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, enc = [], 2278392
    patients = []
    for _ in range(args.rows):
        enc += random.randint(1, 900)
        if patients and random.random() < 0.3:
            pat = random.choice(patients)
        else:
            pat = random.randint(100000, 190000000)
            patients.append(pat)
        rows.append(make_row(enc, pat))
    # dirty rows to exercise validation rules
    rows.append(list(rows[10]))                  # duplicate encounter
    bad = list(rows[20]); bad[SOURCE_COLUMNS.index("time_in_hospital")] = "0"; rows.append(bad)
    bad = list(rows[30]); bad[0] = "?"; rows.append(bad)
    bad = list(rows[40]); bad[SOURCE_COLUMNS.index("readmitted")] = "YES"; rows.append(bad)

    path = out / "diabetic_data.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SOURCE_COLUMNS)
        w.writerows(rows)
        f.write("99999999,123,Caucasian,Female\n")   # malformed line (too few fields)
    (out / "IDs_mapping.csv").write_text(MAPPING_TEXT, encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic rows (+1 malformed line) to {path}")


if __name__ == "__main__":
    main()
