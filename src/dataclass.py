import json
from typing import Dict, List

import torch
from config import DATA_DIR
from utils import Article
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import simple_icd_10 as icd

torch.manual_seed(42)


class Sundhed(Dataset):
    def __init__(self, data: List[Article], tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.article_ids = [article["id_"] for article in self.data]
        self._make_translation_dicts_onehot()
        self._prepare_labels()

    def __getitem__(self, index):

        sample = {key: torch.tensor(val[index]) for key, val in self.inputs.items()}
        sample["chapters"] = self.chapters[index]
        sample["blocks"] = self.blocks[index]
        sample["categories"] = self.categories[index]
        sample["article_ids"] = self.article_ids[index]

        return sample

    def __len__(self):
        return len(self.data)

    def _make_translation_dicts_onehot(self) -> None:
        """Convert from str to int and back"""
        # making sure we always have the same order
        self.unique_chapters = sorted(
            list(set([item for sublist in self.data for item in sublist["chapters"]]))
        )

        self.unique_blocks = sorted(
            list(set([item for sublist in self.data for item in sublist["blocks"]]))
        )
        self.unique_categories = sorted(
            list(set([item for sublist in self.data for item in sublist["categories"]]))
        )

        # save count of chapters, blocks and categories
        self.n_chapters = len(self.unique_chapters)
        self.n_blocks = len(self.unique_blocks)
        self.n_categories = len(self.unique_categories)

        blocks_in_chapters: List[List[int]] = [
            icd.get_children(chapter) for chapter in self.unique_chapters
        ]
        pruned_blocks_in_chapters: List[int] = []
        for chapter_blocks in blocks_in_chapters:
            chapter_blocks_in_dataset = [
                value for value in chapter_blocks if value in self.unique_blocks
            ]
            pruned_blocks_in_chapters.append(chapter_blocks_in_dataset)
        flattend_pruned_blocks_in_chapters = [
            value for sublist in pruned_blocks_in_chapters for value in sublist
        ]

        self.pruned_blocks_in_chapters = pruned_blocks_in_chapters

        # save the mappings
        self.chapter_to_int = {
            chapter: i for i, chapter in enumerate(self.unique_chapters)
        }
        self.block_to_int = {
            block: i for i, block in enumerate(flattend_pruned_blocks_in_chapters)
        }  # this is just to make sure I have the right order in the blocks
        self.category_to_int = {
            category: i for i, category in enumerate(self.unique_categories)
        }
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
