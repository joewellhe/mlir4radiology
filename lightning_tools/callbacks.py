import os
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import TQDMProgressBar

import logging

class CustomProgressBar(TQDMProgressBar):
    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        for key in list(items.keys()):
            if "loss" not in key:
                items.pop(key)
        return items


def add_callbacks(args):
    log_dir = args.savedmodel_path
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="{epoch}-{step}",
        save_top_k=-1,
        every_n_train_steps=args.every_n_train_steps,
        save_last=False,
        save_weights_only=False
    )
    
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    # 显式添加 TQDM 进度条，refresh_rate=10 表示每 10 个 step 刷新一次
    # 这能确保即使在日志文件中也能看到进度更新，同时避免文件过大
    progress_bar = CustomProgressBar(refresh_rate=1)
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {
        "callbacks": [checkpoint_callback, lr_monitor_callback, progress_bar],
        "loggers": [csv_logger, tb_logger]
    }
    return to_returns
