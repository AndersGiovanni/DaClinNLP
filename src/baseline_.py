import json
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from src.baseline.dataclass import Sundhed
from src.config import DATA_DIR
from src.utils import Article, get_relevant_data
from transformers import AutoTokenizer
import transformers


class ICDBert(torch.nn.Module):
    def __init__(self, model_name: str, n_chapters: int, n_blocks: int) -> None:
        super(ICDBert, self).__init__()
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.chapter = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.model.config.hidden_size, out_features=n_chapters),
        )
        self.block = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(self.model.config.hidden_size, out_features=n_blocks),
        )

    def forward(self, input_ids, attention_mask):

        # Get the hidden states from the encoder.
        _, lm_features = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return {
            "chapters": self.chapter(lm_features),
            "blocks": self.block(lm_features),
        }


def criterion(loss_function, outputs, articles, device) -> float:
    loss = 0.0
    for label, output in outputs.items():
        loss += loss_function(output, articles[label].to(device))
    return loss / len(outputs)


def training(model, train_loader, val_loader, device, epochs, lr_rate, weight_decay):

    losses = []
    checkpoint_losses = []
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_rate, weight_decay=weight_decay
    )
    loss_function = torch.nn.BCELoss()

    n_total_steps = len(train_loader) * epochs

    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(train_loader):
            # cast to device
            batch = {k: v.to(device) for k, v in batch.items()}
            # forward pass
            outputs = model(**batch)
            # compute loss
            loss = criterion(loss_function, outputs, batch, device)
            # append loss to list
            losses.append(loss.item())
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
    data: List[Article] = get_relevant_data(data)

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
        "epochs": 10,
        "lr_rate": 1e-5,
        "weight_decay": 0.1,
    }

    # Create model
    model = ICDBert(
        model_name, n_chapters=dataset.n_chapters, n_blocks=dataset.n_blocks
    )

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train model
    training(
        model,
        train_loader,
        val_loader,
        device=device,
        **training_config,
    )
