"""Data module for pytorch lightning"""

import torch
from lightning.pytorch import LightningDataModule
from ocf_datapipes.batch import BatchKey
from ocf_datapipes.load import OpenGSP
from ocf_datapipes.training.pvnet import normalize_gsp
from torch.utils.data import DataLoader
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import FileLister, Zipper

# https://github.com/pytorch/pytorch/issues/973
torch.multiprocessing.set_sharing_strategy("file_system")


class GetNationalPVLive(IterDataPipe):
    """Select national output targets for given times"""

    def __init__(self, gsp_data, times_datapipe):
        """Select national output targets for given times

        Args:
            gsp_data: xarray Dataarray of the national outputs
            times_datapipe: IterDataPipe yeilding arrays of target times.
        """
        self.gsp_data = gsp_data
        self.times_datapipe = times_datapipe

    def __iter__(self):
        gsp_data = self.gsp_data
        for times in self.times_datapipe:
            national_outputs = torch.as_tensor(
                gsp_data.sel(time_utc=times.cpu().numpy().astype("datetime64[s]")).values
            )
            yield national_outputs


class GetBatchTime(IterDataPipe):
    """Extract the valid times from the concurrent sample batch"""

    def __init__(self, sample_datapipe):
        """Extract the valid times from the concurrent sample batch

        Args:
            sample_datapipe: IterDataPipe yeilding concurrent sample batches
        """
        self.sample_datapipe = sample_datapipe

    def __iter__(self):
        for sample in self.sample_datapipe:
            # Times for each GSP in the sample batch should be the same - take first
            id0 = sample[BatchKey.gsp_t0_idx]
            times = sample[BatchKey.gsp_time_utc][0, id0 + 1 :]
            yield times


class PivotDictList(IterDataPipe):
    """Convert list of dicts to dict of lists"""

    def __init__(self, source_datapipe):
        """Convert list of dicts to dict of lists

        Args:
            source_datapipe: Datapipe yielding lists of dicts
        """
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for list_of_dicts in self.source_datapipe:
            keys = list_of_dicts[0].keys()
            batch_dict = {k: [d[k] for d in list_of_dicts] for k in keys}
            yield batch_dict


class DictApply(IterDataPipe):
    """Apply functions to elements of a dictionary and return processed dictionary."""

    def __init__(self, source_datapipe, **transforms):
        """Apply functions to elements of a dictionary and return processed dictionary.

        Args:
            source_datapipe: Datapipe which yields dicts
            **transforms: key-function pairs
        """
        self.source_datapipe = source_datapipe
        self.transforms = transforms

    def __iter__(self):
        for d in self.source_datapipe:
            for key, function in self.transforms.items():
                d[key] = function(d[key])
            yield d


class ZipperDict(IterDataPipe):
    """Yield samples from multiple datapipes as a dict"""

    def __init__(self, **datapipes):
        """Yield samples from multiple datapipes as a dict.

        Args:
            **datapipes: Named datapipes
        """
        self.keys = list(datapipes.keys())
        self.source_datapipes = Zipper(*[datapipes[key] for key in self.keys])

    def __iter__(self):
        for outputs in self.source_datapipes:
            yield {key: value for key, value in zip(self.keys, outputs)}  # noqa: B905


def get_capacity(batch):
    """Extract the capacity from the numpy batch"""
    return batch[BatchKey.gsp_effective_capacity_mwp]


def divide(args):
    """Divide first argument by second"""
    return args[0] / args[1]


class DataModule(LightningDataModule):
    """Datamodule for training pvnet_summation."""

    def __init__(
        self,
        batch_dir: str,
        gsp_zarr_path: str,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
    ):
        """Datamodule for training pvnet_summation.

        Args:
            batch_dir: Path to the directory of pre-saved batches.
            gsp_zarr_path: Path to zarr file containing GSP ID 0 outputs
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.gsp_zarr_path = gsp_zarr_path
        self.batch_size = batch_size
        self.batch_dir = batch_dir

        self._common_dataloader_kwargs = dict(
            batch_size=None,  # batched in datapipe step
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def _get_premade_batches_datapipe(self, subdir, shuffle=False, add_filename=False):
        # Load presaved concurrent sample batches
        file_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)

        if shuffle:
            file_pipeline = file_pipeline.shuffle(buffer_size=10_000)

        file_pipeline = file_pipeline.sharding_filter()

        if add_filename:
            file_pipeline, file_pipeline_copy = file_pipeline.fork(2, buffer_size=5)

        sample_pipeline = file_pipeline.map(torch.load)

        # Find national outout simultaneous to concurrent samples
        gsp_data = (
            next(iter(OpenGSP(gsp_pv_power_zarr_path=self.gsp_zarr_path).map(normalize_gsp)))
            .sel(gsp_id=0)
            .compute()
        )

        sample_pipeline, sample_pipeline_copy = sample_pipeline.fork(2, buffer_size=5)
        times_datapipe = GetBatchTime(sample_pipeline_copy)

        times_datapipe, times_datapipe_copy = times_datapipe.fork(2, buffer_size=5)
        national_targets_datapipe = GetNationalPVLive(gsp_data, times_datapipe_copy)

        times_datapipe, times_datapipe_copy = times_datapipe.fork(2, buffer_size=5)
        national_capacity_datapipe = GetNationalPVLive(
            gsp_data.effective_capacity_mwp, times_datapipe_copy
        )
        sample_pipeline, sample_pipeline_copy = sample_pipeline.fork(2, buffer_size=5)
        gsp_capacity_pipeline = sample_pipeline_copy.map(get_capacity)

        capacity_pipeline = gsp_capacity_pipeline.zip(national_capacity_datapipe).map(divide)

        # Compile the samples
        if add_filename:
            data_pipeline = ZipperDict(
                pvnet_inputs=sample_pipeline,
                effective_capacity=capacity_pipeline,
                national_targets=national_targets_datapipe,
                times=times_datapipe,
                filepath=file_pipeline_copy,
            )
        else:
            data_pipeline = ZipperDict(
                pvnet_inputs=sample_pipeline,
                effective_capacity=capacity_pipeline,
                national_targets=national_targets_datapipe,
                times=times_datapipe,
            )

        if self.batch_size is not None:
            data_pipeline = PivotDictList(data_pipeline.batch(self.batch_size))
            data_pipeline = DictApply(
                data_pipeline,
                effective_capacity=torch.stack,
                national_targets=torch.stack,
                times=torch.stack,
            )

        return data_pipeline

    def train_dataloader(self, shuffle=True, add_filename=False):
        """Construct train dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "train", shuffle=shuffle, add_filename=add_filename
        )
        return DataLoader(datapipe, shuffle=shuffle, **self._common_dataloader_kwargs)

    def val_dataloader(self, shuffle=False, add_filename=False):
        """Construct val dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "val", shuffle=shuffle, add_filename=add_filename
        )
        return DataLoader(datapipe, shuffle=shuffle, **self._common_dataloader_kwargs)

    def test_dataloader(self):
        """Construct test dataloader"""
        raise NotImplementedError


class PVNetPresavedDataModule(LightningDataModule):
    """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation."""

    def __init__(
        self,
        batch_dir: str,
        batch_size=16,
        num_workers=0,
        prefetch_factor=None,
    ):
        """Datamodule for loading pre-saved PVNet predictions to train pvnet_summation.

        Args:
            batch_dir: Path to the directory of pre-saved batches.
            batch_size: Batch size.
            num_workers: Number of workers to use in multiprocess batch loading.
            prefetch_factor: Number of data will be prefetched at the end of each worker process.
        """
        super().__init__()
        self.batch_size = batch_size
        self.batch_dir = batch_dir

        self._common_dataloader_kwargs = dict(
            batch_size=None,  # batched in datapipe step
            sampler=None,
            batch_sampler=None,
            num_workers=num_workers,
            collate_fn=None,
            pin_memory=False,
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=prefetch_factor,
            persistent_workers=False,
        )

    def _get_premade_batches_datapipe(self, subdir, shuffle=False):
        # Load presaved concurrent sample batches
        file_pipeline = FileLister(f"{self.batch_dir}/{subdir}", masks="*.pt", recursive=False)

        if shuffle:
            file_pipeline = file_pipeline.shuffle(buffer_size=10_000)

        sample_pipeline = file_pipeline.sharding_filter().map(torch.load)

        if self.batch_size is not None:
            batch_pipeline = PivotDictList(sample_pipeline.batch(self.batch_size))
            batch_pipeline = DictApply(
                batch_pipeline,
                effective_capacity=torch.stack,
                pvnet_outputs=torch.stack,
                national_targets=torch.stack,
                times=torch.stack,
            )

        return batch_pipeline

    def train_dataloader(self):
        """Construct train dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "train",
            shuffle=True,
        )
        return DataLoader(datapipe, shuffle=True, **self._common_dataloader_kwargs)

    def val_dataloader(self):
        """Construct val dataloader"""
        datapipe = self._get_premade_batches_datapipe(
            "val",
            shuffle=False,
        )
        return DataLoader(datapipe, shuffle=False, **self._common_dataloader_kwargs)

    def test_dataloader(self):
        """Construct test dataloader"""
        raise NotImplementedError
