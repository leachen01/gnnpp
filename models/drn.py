import pytorch_lightning as L
import torch.nn as nn
from models.model_utils import EmbedStations
from models.loss import NormalCRPS
import torch


class DRN(L.LightningModule):
    """DRN from Rasp and Lerch - 2018 - Neural Networks for Postprocessing Ensemble Weathe.pdf"""

    def __init__(self, embedding_dim, in_channels, hidden_channels, optimizer_class, optimizer_params) -> None:
        super(DRN, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params

        self.embedding = EmbedStations(num_stations_max=122, embedding_dim=embedding_dim)

        self.linear = nn.ModuleList()
        for hidden_size in self.hidden_channels:
            self.linear.append(nn.Linear(in_features=in_channels, out_features=hidden_size))
            in_channels = hidden_size
        self.last_linear_mu = nn.Linear(in_features=in_channels, out_features=1)
        self.last_linear_sigma = nn.Linear(in_features=in_channels, out_features=1)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.loss_fn = NormalCRPS()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.linear:
            x = layer(x)
            x = self.relu(x)
        mu = self.last_linear_mu(x)  # Last Layer without ReLU
        sigma = self.softplus(self.last_linear_sigma(x))
        res = torch.cat([mu, sigma], dim=1)
        return res

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn.crps(mu_sigma=y_hat, y=y.flatten())
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer_class(self.parameters(), **self.optimizer_params)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn.crps(mu_sigma=y_hat, y=y.flatten())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn.crps(mu_sigma=y_hat, y=y.flatten())
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat
