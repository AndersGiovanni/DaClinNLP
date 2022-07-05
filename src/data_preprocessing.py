import csv
import json
from src.config import DATA_DIR


if __name__ == "__main__":

    data_with_icd = {}
    data_with_no_icd = {}
    with open(DATA_DIR / "processed_and_combined_data.json", "r") as f:
        data = json.load(f)

        for k, v in data[0].items():
            if "ICD" in v["MetaTags"]:
                data_with_icd[k] = v
            else:
                data_with_no_icd[k] = v

    icd_codes = {}
    with open(DATA_DIR / "icd_codes_danish/d_diagnosis_codes.csv") as codes:
        f = csv.reader(codes, delimiter=";")
        for i in f:
            icd_code, icd_description = i[0], i[1]
            if icd_code[1:] == "":
                continue
            icd_codes[icd_code[1:]] = icd_description

    a = 1
