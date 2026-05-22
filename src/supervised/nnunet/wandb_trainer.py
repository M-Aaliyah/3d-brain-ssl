from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import wandb


class nnUNetTrainerWandB(nnUNetTrainer):
    """nnUNetTrainer subclass that integrates Weights & Biases for experiment tracking."""
    def on_train_start(self):
        super().on_train_start()

        wandb.init(
            project="3d-brain-ssl",
            name=self.experiment_name,
            config={
                "configuration": self.configuration_name,
                "fold": self.fold,
            },
        )

    def on_epoch_end(self):
        super().on_epoch_end()

        wandb.log({
            "epoch": self.current_epoch,
            "train_loss": self.logger.my_fantastic_logging["train_losses"][-1],
            "val_dice": self.logger.my_fantastic_logging["mean_fg_dice"][-1],
        })

    def on_train_end(self):
        wandb.finish()
        super().on_train_end()