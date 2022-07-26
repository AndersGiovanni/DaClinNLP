from typing import Dict, List

import simple_icd_10 as icd
import torch
from typing_extensions import TypedDict

Article = TypedDict(
    "Article",
    {
        "id_": str,
        "title": str,
        "description": str,
        "body": str,
        "chapters": List[str],
        "blocks": List[str],
        "categories": List[str],
    },
)


def get_danish_icd_codes() -> Dict[str, str]:
    icd_codes = {}
    with open("../../data/icd_codes_danish/d_diagnosis_codes.csv") as codes:
        f = codes.readlines()
        for i in f:
            icd_code, icd_description = i.strip().split(";")
            icd_codes[icd_code[2:-1]] = icd_description[1:-1]
    return icd_codes


def get_relevant_data(data: Dict) -> List[Article]:

    relevant_data: List[Article] = []
    for id_, article_data in data.items():

        title_of_first_paragraph: str = list(article_data["texts"].keys())[0]

        relevant_data.append(
            Article(
                id_=id_,
                title=article_data["MetaTags"]["title"],
                description=article_data["MetaTags"]["description"],
                body=article_data["texts"][title_of_first_paragraph],
                chapters=article_data["MetaTags"]["ICD_details"]["chapters"],
                blocks=article_data["MetaTags"]["ICD_details"]["blocks"],
                categories=article_data["MetaTags"]["ICD_details"]["categories"],
            )
        )
    return relevant_data


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.has_mps:
        return "mps"
    else:
        return "cpu"


def convert_results_from_int_to_string(results: Dict, dataset) -> List[Dict]:
    """Convert the results from int to string. The dataset is only for the conversion. Ugly piece of code, sorry."""
    output = []
    for article_id, article_results in results.items():
        results_object = {"id_": article_id}
        for result_id, result in article_results.items():
            if "chapters" in result_id:
                results_object[result_id] = [dataset.int_to_chapter[i] for i in result]
            else:
                results_object[result_id] = [dataset.int_to_block[i] for i in result]
        output.append(results_object)
    return output


if __name__ == "__main__":
    pass
