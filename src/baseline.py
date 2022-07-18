import json
from typing import Dict, List


import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import DATA_DIR
from src.utils import get_relevant_data, Article


class Sundhed(Dataset):
    def __init__(self, data: List[Article], tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self._make_translation_dicts_onehot()
        self._prepare_labels()

    def _make_translation_dicts_onehot(self) -> None:
        """Convert from str to int and back"""
        chapters = list(
            set([item for sublist in self.data for item in sublist["chapters"]])
        )

        blocks = list(
            set([item for sublist in self.data for item in sublist["blocks"]])
        )
        categories = list(
            set([item for sublist in self.data for item in sublist["categories"]])
        )

        self.chapter_to_int = {chapter: i for i, chapter in enumerate(chapters)}
        self.block_to_int = {block: i for i, block in enumerate(blocks)}
        self.category_to_int = {category: i for i, category in enumerate(categories)}
        self.int_to_chapter = {i: chapter for chapter, i in self.chapter_to_int.items()}
        self.int_to_block = {i: block for block, i in self.block_to_int.items()}
        self.int_to_category = {
            i: category for category, i in self.category_to_int.items()
        }

    def _tokenize_data(
        self, text_fields_to_use: List[str] = ["title", "description", "body"]
    ) -> None:
        """Tokenize the data. At first select the text fields to use and then tokenize them."""

        texts: List[str] = []
        for article in self.data:
            text = ""
            for text_field in text_fields_to_use:
                text += f"{article[text_field]}. "
            texts.append(text)

        inputs: Dict[str, torch.tensor] = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt"
        )

        # making sure the sizes are correct
        assert len(inputs["input_ids"]) == len(
            self.data
        ), "Number of inputs and data do not match"

        self.inputs = inputs

    def _prepare_labels(self):
        """One-hot the labels"""
        self.chapters = torch.zeros(len(self.data), len(self.chapter_to_int))
        self.blocks = torch.zeros(len(self.data), len(self.block_to_int))
        self.categories = torch.zeros(len(self.data), len(self.category_to_int))

        for i, article in enumerate(self.data):
            for chapter in article["chapters"]:
                self.chapters[i][self.chapter_to_int[chapter]] = 1
            for block in article["blocks"]:
                self.blocks[i][self.block_to_int[block]] = 1
            for category in article["categories"]:
                self.categories[i][self.category_to_int[category]] = 1

    def __getitem__(self, index):

        sample = {key: torch.tensor(val[index]) for key, val in self.inputs.items()}
        sample["chapters"] = self.chapters[index]
        sample["blocks"] = self.blocks[index]
        sample["categories"] = self.categories[index]

        return sample

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    with open(DATA_DIR / "icd_coding.json", "r") as f:
        data: Dict = json.load(f)[0]

    data: List[Article] = get_relevant_data(data)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = Sundhed(data, tokenizer)

    dataset._tokenize_data(["title"])

    a = 1
