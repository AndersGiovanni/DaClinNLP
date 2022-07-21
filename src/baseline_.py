import json

# Ignore excessive warnings
import logging
from typing import Dict, List

import numpy as np
import simple_icd_10 as icd
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score

from dataclass import Sundhed
from config import DATA_DIR
from utils import Article, get_device, get_relevant_data

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb


class ICDBert(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        n_chapters: int,
        n_blocks: int,
        blocks_in_chapters: List[List[str]],
    ) -> None:
        super(ICDBert, self).__init__()
        # base model
        self.model = transformers.AutoModel.from_pretrained(model_name)
        # chapter classification
        self.chapter = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=self.model.config.dim, out_features=n_chapters),
        )
        # block classification
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Dropout(0.3),
                    torch.nn.Linear(
                        in_features=self.model.config.dim,
                        out_features=len(blocks_in_chapters[i]),
                    ),
                )
                for i in range(len(blocks_in_chapters))
            ]
        )
        # sigmoid activation
        self.sigmoid = torch.nn.Sigmoid()
        self.n_blocks = n_blocks
        self.n_chapters = n_chapters
        self.blocks_in_chapters = blocks_in_chapters
        self._get_block_offsets()

    def _get_block_offsets(self) -> None:
        """We only want to insert the blocks in the correct chapter, so we need to know the offset of each block."""
        self.block_offsets = {}
        start, end = 0, 0
        for idx, block_in_chapter in enumerate(self.blocks_in_chapters):
            end += len(block_in_chapter)
            self.block_offsets[idx] = (start, end)
            start = end

    def forward(self, input_ids, attention_mask):

        # Get the hidden states from the encoder.
        last_hidden_state = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch_size, sequence_length, hidden_size)
        pooled_output = last_hidden_state[:, 0]  # (batch_size, hidden_size)

        # chapter classification
        chapter_logits = self.chapter(pooled_output)  # (batch_size, n_chapters)
        chapter_probs = self.sigmoid(chapter_logits)  # (batch_size, n_chapters)

        # block classification
        # This is a conditional classification based on the output of the chapter classification.
        block_probs_batch = torch.zeros(
            (len(input_ids), self.n_blocks)
        )  # Empty tensor (batch_size, n_blocks)
        positive_predictions_chapter = (
            chapter_probs >= 0.5
        )  # Get all chapter probs >= 0.5
        positive_predictions_idx: List[
            List[str]
        ] = (
            positive_predictions_chapter.nonzero().tolist()
        )  # Get all indices where chapter probs >= 0.5
        for (
            pos_prediction_list
        ) in (
            positive_predictions_idx
        ):  # For each chapter with a positive prediction, perhaps more than one
            (
                row_idx,
                col_idx,
            ) = pos_prediction_list  # row is the batch index, col is the chapter index
            block_logits = self.blocks[col_idx](
                pooled_output[row_idx]
            )  # Get the logits for the block
            block_probs = self.sigmoid(
                block_logits
            )  # Get the probabilities for the block
            block_probs_batch[
                row_idx,
                self.block_offsets[col_idx][0] : self.block_offsets[col_idx][1],
            ] = block_probs  # Insert the probabilities for the block

        return {
            "chapters": chapter_probs,
            "blocks": block_probs_batch,
        }


def criterion(loss_function, outputs, articles, device) -> float:
    loss = torch.tensor(0.0).to(device)
    # accumulate loss for both chapters and blocks
    for label, output in outputs.items():
        loss += loss_function(output, articles[label].to(device))
    return loss


# def training(model, train_loader, device, epochs, lr_rate, weight_decay):
def training(args, model, device, train_loader, optimizer, epoch):

    # cast model to device
    model = model.to(device)

    loss_function = torch.nn.BCELoss()

    model.train()

    for batch_idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        # get input and labels and cast to device
        batch_input = {k: batch[k].to(device) for k in ["input_ids", "attention_mask"]}
        batch_labels = {k: batch[k].to(device) for k in ["chapters", "blocks"]}

        # Resent gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(**batch_input)

        # compute loss
        loss = criterion(loss_function, outputs, batch_labels, device)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()


def test(args: str, model: torch.nn.Module, device, test_loader):
    model.eval()
    test_loss = 0.0
    predictions_and_labels = {
        "chapters_preds": [],
        "blocks_preds": [],
        "chapters_labels": [],
        "blocks_labels": [],
    }
    with torch.no_grad():
        for batch in test_loader:
            batch_input = {
                k: batch[k].to(device) for k in ["input_ids", "attention_mask"]
            }
            batch_labels = {k: batch[k].to(device) for k in ["chapters", "blocks"]}
            outputs = model(**batch_input)
            test_loss += criterion(torch.nn.BCELoss(), outputs, batch_labels, device)
            for label, output in outputs.items():
                output = output > 0.5
                (
                    predictions_and_labels[label + "_preds"]
                    + output.cpu().flatten().tolist()
                )
                (
                    predictions_and_labels[label + "_labels"]
                    + batch_labels[label].cpu().flatten().tolist()
                )

    wandb.log(
        {
            f"{args} loss": test_loss / len(test_loader.dataset),
            f"{args} Precision chapters": precision_score(
                predictions_and_labels["chapters_preds"],
                predictions_and_labels["chapters_labels"],
            ),
            f"{args} Recall chapters": recall_score(
                predictions_and_labels["chapters_preds"],
                predictions_and_labels["chapters_labels"],
            ),
            f"{args} F1 chapters": f1_score(
                predictions_and_labels["chapters_preds"],
                predictions_and_labels["chapters_labels"],
            ),
            f"{args} Precision blocks": precision_score(
                predictions_and_labels["blocks_preds"],
                predictions_and_labels["blocks_labels"],
            ),
            f"{args} Recall blocks": recall_score(
                predictions_and_labels["blocks_preds"],
                predictions_and_labels["blocks_labels"],
            ),
            f"{args} F1 blocks": f1_score(
                predictions_and_labels["blocks_preds"],
                predictions_and_labels["blocks_labels"],
            ),
            f"{args} Flat total precision": precision_score(
                predictions_and_labels["chapters_preds"]
                + predictions_and_labels["blocks_preds"],
                predictions_and_labels["chapters_labels"]
                + predictions_and_labels["blocks_labels"],
            ),
            f"{args} Flat total recall": recall_score(
                predictions_and_labels["chapters_preds"]
                + predictions_and_labels["blocks_preds"],
                predictions_and_labels["chapters_labels"]
                + predictions_and_labels["blocks_labels"],
            ),
            f"{args} Flat total F1": f1_score(
                predictions_and_labels["chapters_preds"]
                + predictions_and_labels["blocks_preds"],
                predictions_and_labels["chapters_labels"]
                + predictions_and_labels["blocks_labels"],
            ),
        }
    )


if __name__ == "__main__":

    # WandB – Initialize a new run
    wandb.init(entity="hrmussa", project="icd_coding")
    # wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config  # Initialize config
    config.batch_size = 16  # input batch size for training (default: 64)
    config.test_batch_size = 8  # input batch size for testing (default: 1000)
    config.epochs = 1  # number of epochs to train (default: 10)
    config.lr = 1e-5  # learning rate (default: 0.01)
    config.weight_decay = 0.1  # weight decay (default: 0.0)
    # config.momentum = 0.1  # SGD momentum (default: 0.5)
    config.no_cuda = True  # disables CUDA training
    config.seed = 42  # random seed (default: 42)
    config.log_interval = 5  # how many batches to wait before logging training status

    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    np.random.seed(config.seed)  # numpy random seed
    # torch.backends.cudnn.deterministic = True

    model_name = "Geotrend/distilbert-base-da-cased"

    with open(DATA_DIR / "icd_coding.json", "r") as f:
        data: Dict = json.load(f)[0]

    # Get relevant data
    data: List[Article] = get_relevant_data(data)[:40]

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataset
    dataset = Sundhed(data, tokenizer)

    # Get the amount of unique blocks in each chaper. This is for the model construction.
    pruned_blocks_in_chapters: List[List[str]] = dataset.pruned_blocks_in_chapters

    # Select which fields to use
    text_fields_to_use = ["title", "description", "body"]
    dataset._tokenize_data(["title"])

    # Make train test split
    train_pct: float = 0.8
    dataset_length: int = dataset.__len__()
    train_length: int = int(dataset_length * train_pct)
    val_length: int = int((dataset_length - train_length) / 2)
    test_length: int = val_length

    # check if the lengths are correct
    if (train_length + 2 * val_length) != dataset_length:
        test_length = val_length + 1

    # Split dataset
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_length, val_length, test_length],
        generator=torch.Generator().manual_seed(42),
    )

    # Create dataloaders
    train_loader: DataLoader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_loader: DataLoader = DataLoader(
        val_set, batch_size=config["test_batch_size"], shuffle=True
    )
    test_loader: DataLoader = DataLoader(
        test_set, batch_size=config["test_batch_size"], shuffle=True
    )

    # Create model
    model = ICDBert(
        model_name,
        n_chapters=dataset.n_chapters,
        n_blocks=dataset.n_blocks,
        blocks_in_chapters=pruned_blocks_in_chapters,
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # get device
    device = torch.device(get_device())
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    wandb.watch(model, log="all", log_graph=True, log_freq=5)

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        training(config, model, device, train_loader, optimizer, epoch)
        test("Val", model, device, val_loader)

    # Test
    test("Test", model, device, test_loader)

    # Save model
    wandb.save("model.h5")

    wandb.finish()
