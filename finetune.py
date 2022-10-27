from model_pl import TripletLossInceptionResnetV1
from dataset import TripletDataModule
import pytorch_lightning as pl
from omegaconf import OmegaConf

if __name__ == '__main__':
    config_path = 'config/finetune.yaml'

    config = OmegaConf.load(config_path)
    dataset_config = config.dataset
    model_config = config.model
    trainer_config = config.trainer

    model = TripletLossInceptionResnetV1(model_config)
    dataset = TripletDataModule(dataset_config)

    lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='step')
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        save_top_k=10,
        filename='{epoch}-{val_loss:.3f}'
    )

    trainer = pl.Trainer(
        **trainer_config,
        callbacks=[lr_monitor_cb, checkpoint_cb],
    )

    trainer.fit(model, dataset)