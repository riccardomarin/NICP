import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataloader import default_collate

#from nn_core.common import PROJECT_ROOT
from pathlib import Path
from lvd_templ.paths import neutral_smpl_path, home_dir
from nn_core.nn_types import Split

## CHANGE PATH TO THE CURRENT FOLDER
PROJECT_ROOT = Path(home_dir)

pylogger = logging.getLogger(__name__)

class MetaData:
    def __init__(self, class_vocab: Mapping[str, int]):
        self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path) -> None:
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        pylogger.debug(f"Loading MetaData from '{src_path}'")


def collate_fn(samples: List, split: Split, metadata: MetaData):
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        # example
        overfit: bool,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = False

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        # example
        self.overfit: bool = overfit

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(class_vocab=self.train_dataset.class_vocab)

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):

        #################################################
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            # Instantiate training dataset
            train_data = hydra.utils.instantiate(self.datasets.train, mode="train")

            if self.overfit:
                val_data = hydra.utils.instantiate(self.datasets.train, mode="train")
                self.val_datasets = val_data
            else:
                val_data = hydra.utils.instantiate(self.datasets.train, mode="val")

            self.train_dataset = train_data
            self.val_dataset = val_data

        if self.overfit:
            self.datasets.train["augm_noise"] = False
            test_data = hydra.utils.instantiate(self.datasets.train, mode="train")
            self.test_datasets = test_data
        else:
            if stage is None or stage == "test":
                test_data = hydra.utils.instantiate(self.datasets.train, mode="test")
                self.test_datasets = test_data

        
        #################################################

    def train_dataloader(self) -> DataLoader:
        if self.overfit:
            shuff = False
        else:
            shuff = True

        return DataLoader(
            self.train_dataset,
            shuffle=shuff,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return DataLoader(
            self.test_datasets,
            shuffle=True,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf_ifnet"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)


if __name__ == "__main__":
    main()
