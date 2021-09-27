import argparse
import os
import pathlib
from typing import Callable, List, Tuple

import h5py
import numpy as np
from nvidia.dali.pipeline import Pipeline

import data as datam  # "datam" = "data module"
import utils


@utils.ArgumentParser.register_extension("HDF5 Pipeline")
def add_h5_argument_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--use_h5", action="store_true",
                        help="Whether to use HDF5 data.")


def stage_files(dist_desc: utils.DistributedEnvDesc,
                data_dir: pathlib.Path,
                output_dir: pathlib.Path,
                data_filenames: List[str],
                label_filenames: List[str],
                shard_mult: int) -> Tuple[List[str], List[str], Callable]:
    number_of_nodes = dist_desc.size // dist_desc.local_size // shard_mult
    current_node = dist_desc.rank // dist_desc.local_size // shard_mult
    all_indices = np.arange(len(data_filenames))

    files_per_node = len(data_filenames) // number_of_nodes
    per_node_indices = all_indices[
        current_node * files_per_node:(current_node+1) * files_per_node]
    per_node_data_filenames = data_filenames[
        current_node * files_per_node:(current_node+1) * files_per_node]
    per_node_label_filenames = label_filenames[
        current_node * files_per_node:(current_node+1) * files_per_node]

    os.makedirs(output_dir, exist_ok=True)
    # print("started staging files")
    h5_path = data_dir / (data_dir.name + '.h5')

    with h5py.File(h5_path, 'r') as h5_file:
        data_dset = h5_file['data']
        label_dset = h5_file['label']
        copied_files = 0
        for (i, data, label) in zip(
                per_node_indices[
                    dist_desc.local_rank::dist_desc.local_size],
                per_node_data_filenames[
                    dist_desc.local_rank::dist_desc.local_size],
                per_node_label_filenames[
                    dist_desc.local_rank::dist_desc.local_size],
        ):
            np_data = data_dset[i]
            np.save(output_dir / data, np_data)

            np_label = label_dset[i]
            np.save(output_dir / label, np_label)

            copied_files += 1

    # print(f"Node {current_node}, process {dist_desc.local_rank}, "
    #       f"dataset contains {len(data_filenames)} samples, "
    #       f"per node {len(per_node_data_filenames)}, copied {copied_files}")

    dist_desc.comm.Barrier()
    return per_node_data_filenames, per_node_label_filenames


class H5CosmoDataset(datam.CosmoDataset):
    def _construct_pipeline(self, data_dir: pathlib.Path,
                            batch_size: int,
                            n_samples: int = -1,
                            prestage: bool = True,
                            shard: datam.ShardType = "none",
                            shuffle: bool = False,
                            preshuffle: bool = False,
                            shard_mult: int = 1) -> Tuple[Pipeline, int]:
        # We need this to always be `True`.
        prestage = True

        data_filenames = datam._load_file_list(data_dir, "files_data.lst")
        label_filenames = datam._load_file_list(data_dir, "files_label.lst")

        if n_samples > 0:
            data_filenames = data_filenames[:n_samples]
            label_filenames = label_filenames[:n_samples]
        n_samples = len(data_filenames) * self.samples_per_file

        if preshuffle:
            preshuffle_permutation = np.ascontiguousarray(
                np.random.permutation(n_samples))
            self.dist.comm.Bcast(preshuffle_permutation, root=0)

            # FIXME This only works if DALI does not care about the
            #       filename list order.
            #       If that is not the case, we could instead shuffle the
            #       indices querying the HDF5 file.
            #       Finally, since the data is already pre-shuffled,
            #       we can also just omit this.
            data_filenames = \
                list(np.array(data_filenames)[preshuffle_permutation])
            label_filenames = \
                list(np.array(label_filenames)[preshuffle_permutation])

        if shard == "local":
            if shard_mult == 1:
                shard_id = self.dist.local_rank
                num_shards = self.dist.local_size
            else:
                node_in_chunk = self.dist.node % shard_mult
                num_shards = self.dist.local_size * shard_mult
                shard_id = (
                    node_in_chunk
                    * self.dist.local_size + self.dist.local_rank
                )
        elif shard == "global":
            shard_id, num_shards = self.dist.rank, self.dist.size
        else:
            shard_id, num_shards = 0, 1

        def pipeline_builder():
            data_filenames_ = data_filenames
            label_filenames_ = label_filenames
            data_dir_ = data_dir

            if prestage:
                output_path = \
                    pathlib.Path("/tmp", "dataset") / data_dir.parts[-1]
                data_filenames_, label_filenames_ = stage_files(
                    self.dist,
                    data_dir,
                    output_path,
                    data_filenames,
                    label_filenames,
                    shard_mult,
                )
                data_dir_ = output_path

            return datam.get_dali_pipeline(
                data_dir_,
                data_filenames_,
                label_filenames_,
                dont_use_mmap=not self.use_mmap,
                shard_id=shard_id,
                num_shards=num_shards,
                apply_log=self.apply_log,
                batch_size=batch_size,
                dali_threads=self.threads,
                device_id=self.dist.local_rank,
                shuffle=shuffle,
                data_layout=self.data_layout,
                sample_shape=self.data_shapes[0],
                target_shape=self.data_shapes[1],
                seed=self.seed,
            )

        return (pipeline_builder,
                n_samples)


# We could also rewrite this method in `data.py` to pick the dataset
# class depending on the `args`. However, this is more forward-compatible.
def get_rec_iterators(
        args: argparse.Namespace,
        dist_desc: utils.DistributedEnvDesc,
) -> Tuple[Callable, int, int]:
    cosmoflow_dataset = H5CosmoDataset(
        args.data_root_dir,
        dist=dist_desc,
        use_mmap=args.dali_use_mmap,
        apply_log=args.apply_log_transform,
        dali_threads=args.dali_num_threads,
        data_layout=args.data_layout,
        seed=args.seed,
    )
    train_iterator_builder, training_steps, training_samples = \
        cosmoflow_dataset.training_dataset(
            args.training_batch_size,
            args.shard_type,
            args.shuffle,
            args.preshuffle,
            args.training_samples,
            args.data_shard_multiplier,
            args.prestage,
        )
    val_iterator_builder, val_steps, val_samples = \
        cosmoflow_dataset.validation_dataset(
            args.validation_batch_size, True, args.validation_samples)

    # MLPerf logging of batch size, and number of samples used in training
    utils.logger.event(key=utils.logger.constants.GLOBAL_BATCH_SIZE,
                       value=args.training_batch_size*dist_desc.size)
    utils.logger.event(
        key=utils.logger.constants.TRAIN_SAMPLES, value=training_samples)
    utils.logger.event(
        key=utils.logger.constants.EVAL_SAMPLES, value=val_samples)

    def iterator_builder():
        utils.logger.start(key='staging_start')
        train_iterator = train_iterator_builder()
        val_iterator = val_iterator_builder()
        utils.logger.end(key='staging_stop')

        return train_iterator, val_iterator

    return (iterator_builder,
            training_steps,
            val_steps)
