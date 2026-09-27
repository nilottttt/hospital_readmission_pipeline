# Dataset source and access instructions

| Item | Value |
|---|---|
| Name | Diabetes 130-US Hospitals for Years 1999–2008 |
| Publisher | UCI Machine Learning Repository (dataset id 296) |
| Page | https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008 |
| Direct download used by the pipeline | https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip |
| Licence | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Original study | Strack B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International, 2014, 781670. |
| Contents | `diabetic_data.csv` (101,766 encounters × 50 columns), `IDs_mapping.csv` (code descriptions) |
| Period | 10 years (1999–2008), 130 US hospitals and integrated delivery networks |
| Privacy | De-identified by the publisher: no names, addresses, dates or real record numbers. `patient_nbr` and `encounter_id` are arbitrary numbers, and the pipeline replaces them with salted hashes anyway. |

## Inclusion criteria used by the publisher
Each row is one inpatient encounter of a patient with any kind of diabetes, lasting 1–14 days, with laboratory tests and medications given during the stay.

## How the pipeline gets the data
1. **Automatic (default):** the `extract` task downloads the zip from the URL above (3 attempts, increasing wait between them), stores it unchanged in `data/raw/<run_id>/source.zip` and extracts the two CSVs next to it.
2. **Manual fallback:** if the download is blocked (for example by a company firewall), download the zip in a browser from the dataset page ("Download" button) and put it in `data/manual/`. The next run will use it automatically and record `status = manual_fallback` in `audit.ingestion_log`.

The URL can be changed without touching the code with `DATA_SOURCE_URL` in `.env`.
