from turtle import forward
import torch
from typing import Tuple
import pytorch_lightning as pl
from facenet_pytorch import InceptionResnetV1
from loss import TripletLoss
from omegaconf import DictConfig

class TripletLossInceptionResnetV1(pl.LightningModule):
    def __init__(self, config: DictConfig = None):
        super(TripletLossInceptionResnetV1, self).__init__()
        self.save_hyperparameters()

        self.resnet = InceptionResnetV1(pretrained='vggface2')
        self.loss = TripletLoss(0.2)
        
    def training_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        loss = self.shared_step(batch)
        self.log('val_loss', loss)
        return loss

    def shared_step(
        self,
        batch: Tuple[torch.Tensor],
    ) -> torch.Tensor:

        embedding = [self.resnet(x) for x in batch]
        loss = self.loss(*embedding)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=2.0)
        return optim

    def forward(self, x):
        return self.resnet(x)