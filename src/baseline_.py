import json
from typing import Dict, List

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.baseline.dataclass import Sundhed
from src.config import DATA_DIR
from src.utils import Article, get_device, get_relevant_data


class ICDBert(torch.nn.Module):
    def __init__(self, model_name: str, n_chapters: int, n_blocks: int) -> None:
        super(ICDBert, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.chapter = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=self.model.config.dim, out_features=n_chapters),
        )
        self.block = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(in_features=self.model.config.dim, out_features=n_blocks),
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):

        # Get the hidden states from the encoder.
        last_hidden_state = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state  # (batch_size, sequence_length, hidden_size)
        pooled_output = last_hidden_state[:, 0]  # (batch_size, hidden_size)

        return {
            "chapters": self.sigmoid(self.chapter(pooled_output)),
            "blocks": self.sigmoid(self.block(pooled_output)),
        }


def criterion(loss_function, outputs, articles, device) -> float:
    loss = 0.0
    for label, output in outputs.items():
        loss += loss_function(output, articles[label].to(device))
    return loss


# def training(model, train_loader, device, epochs, lr_rate, weight_decay):
def training(args, model, device, train_loader, optimizer, epoch):

    # cast model to device
    model = model.to(device)

    loss_function = torch.nn.BCELoss()

    model.train()

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        # get input and labels and cast to device
        batch_input = {
            k: batch[k].to(device) for k in ["input_ids", "attention_mask"]
        }
        batch_labels = {k: batch[k].to(device) for k in ["chapters", "blocks"]}
        # forward pass
        outputs = model(
            **batch_input
        )  # each ouput is size (batch_size, hidden_size, n_chapters) -> (16, 512, 21) for chapters
        # compute loss
        loss = criterion(loss_function, outputs, batch_labels, device)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

            if (i + 1) % (int(n_total_steps / 1)) == 0:
                checkpoint_loss = torch.tensor(losses).mean().item()
                checkpoint_losses.append(checkpoint_loss)
                print(
                    f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {checkpoint_loss:.4f}"
                )


if __name__ == "__main__":

    model_name = "Geotrend/distilbert-base-da-cased"

    with open(DATA_DIR / "icd_coding.json", "r") as f:
        data: Dict = json.load(f)[0]

    # Get relevant data
    data: List[Article] = get_relevant_data(data)[:100]

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataset
    dataset = Sundhed(data, tokenizer)

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
    train_loader: DataLoader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader: DataLoader = DataLoader(val_set, batch_size=8, shuffle=True)
    test_loader: DataLoader = DataLoader(test_set, batch_size=8, shuffle=True)

    training_config = {
        "epochs": 3,
        "lr_rate": 1e-5,
        "weight_decay": 0.1,
    }

    # Create model
    model = ICDBert(
        model_name, n_chapters=dataset.n_chapters, n_blocks=dataset.n_blocks
    )

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config["lr_rate"],
        weight_decay=training_config["weight_decay"],
    )

    # get device
    device = torch.device(get_device())
    print(f"Using device: {device}")

    # Train model
    training(
        model,
        train_loader,
        device=device,
        **training_config,
    )
