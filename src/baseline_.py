import json

# Ignore excessive warnings
import logging
from typing import Dict, List

import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.baseline.dataclass import Sundhed
from src.config import DATA_DIR
from src.utils import Article, get_device, get_relevant_data

logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

import wandb


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


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct_chapters = 0
    correct_blocks = 0
    with torch.no_grad():
        for batch in test_loader:
            batch_input = {
                k: batch[k].to(device) for k in ["input_ids", "attention_mask"]
            }
            batch_labels = {k: batch[k].to(device) for k in ["chapters", "blocks"]}
            outputs = model(**batch_input)
            test_loss += criterion(torch.nn.BCELoss(), outputs, batch_labels, device)
            correct_chapters += (
                (outputs["chapters"] > 0.5)
                .eq(batch_labels["chapters"] > 0.5)
                .sum()
                .item()
            )
            correct_blocks += (
                (outputs["blocks"] > 0.5).eq(batch_labels["blocks"] > 0.5).sum().item()
            )

    wandb.log(
        {
            "Test loss": test_loss / len(test_loader.dataset),
            "Test accuracy chapters": correct_chapters / len(test_loader.dataset),
            "Test accuracy blocks": correct_blocks / len(test_loader.dataset),
        }
    )


if __name__ == "__main__":

    # WandB – Initialize a new run
    wandb.init(entity="hrmussa", project="Just_a_test")
    # wandb.watch_called = False  # Re-run the model without restarting the runtime, unnecessary after our next release

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config  # Initialize config
    config.batch_size = 16  # input batch size for training (default: 64)
    config.test_batch_size = 8  # input batch size for testing (default: 1000)
    config.epochs = 5  # number of epochs to train (default: 10)
    config.lr = 1e-5  # learning rate (default: 0.01)
    config.weight_decay = 0.0  # weight decay (default: 0.0)
    # config.momentum = 0.1  # SGD momentum (default: 0.5)
    config.no_cuda = True  # disables CUDA training
    config.seed = 42  # random seed (default: 42)
    config.log_interval = 5  # how many batches to wait before logging training status

    # Set random seeds and deterministic pytorch for reproducibility
    # random.seed(config.seed)       # python random seed
    torch.manual_seed(config.seed)  # pytorch random seed
    # numpy.random.seed(config.seed) # numpy random seed
    # torch.backends.cudnn.deterministic = True

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

    # Create model
    model = ICDBert(
        model_name, n_chapters=dataset.n_chapters, n_blocks=dataset.n_blocks
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
    wandb.watch(model, log="all", log_graph=True, log_freq=10)

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        training(config, model, device, train_loader, optimizer, epoch)
        test(config, model, device, test_loader)

    # Save model
    wandb.save("model.h5")

    wandb.finish()
