import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pprint import pprint
from configs.config import parser
from dataset.data_module import DataModule
from lightning_tools.callbacks import add_callbacks
from lightning.pytorch import seed_everything
import lightning.pytorch as pl
from model.SCMliR import SCMLIR


# Add project root to sys.path to allow imports like 'evalcap'


def train(args):
    dm = DataModule(args)
    callbacks = add_callbacks(args)

    # 自动修正 DDP 策略以避免 "unused parameters" 错误
    strategy = args.strategy
    if args.strategy == 'ddp':
        strategy = 'ddp_find_unused_parameters_true'

    trainer = pl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        accelerator=args.accelerator,
        precision=args.precision,
        val_check_interval = args.val_check_interval,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches = args.limit_val_batches,
        max_epochs = args.max_epochs,
        num_sanity_val_steps = args.num_sanity_val_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks["callbacks"], 
        logger=callbacks["loggers"]
    )

    if args.ckpt_file is not None:
        model = SCMLIR.load_from_checkpoint(args.ckpt_file, strict=False)
    else:
        model = SCMLIR(args)

    if args.test:
        trainer.test(model, datamodule=dm)
    elif args.validate:
        trainer.validate(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)

def main():
    args = parser.parse_args()
    os.makedirs(args.savedmodel_path, exist_ok=True)
    pprint(vars(args))
    seed_everything(42, workers=True)
    train(args)


if __name__ == '__main__':
    main()