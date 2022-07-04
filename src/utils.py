

from typing import Dict, List
import simple_icd_10 as icd


def get_danish_icd_codes() -> Dict[str, str]:
    icd_codes = {}
    with open('../../data/icd_codes_danish/d_diagnosis_codes.csv') as codes:
        f = codes.readlines()
        for i in f:
            icd_code, icd_description = i.strip().split(';')
            icd_codes[icd_code[2:-1]] = icd_description[1:-1]
    return icd_codes


def get_icd_path(icd_codes_dict: Dict[str, str], icd_code:str) -> List[str]:

    icd_path = []
    


if __name__ == '__main__':

    get_icd_path({}, 's')