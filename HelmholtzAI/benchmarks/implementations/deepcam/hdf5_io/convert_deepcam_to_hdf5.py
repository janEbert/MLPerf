import os
import time

import h5py
from mpi4py import MPI
import numpy as np

preshuffle = True
seed = 42

project_name = 'hai_mlperf'
# Try to use CSCRATCH, fall back to SCRATCH.
using_cscratch = ('CSCRATCH_' + project_name) in os.environ
project_dir = os.getenv(
    'CSCRATCH_' + project_name,
    os.getenv('SCRATCH_' + project_name),
)

root_dir = os.path.join(project_dir, "deepcam_v1.0")
out_dir = os.path.join(project_dir, "deepcam_hdf5")
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(out_dir, exist_ok=True)

np.random.seed(seed)

print("this is the h5py file we are using:", h5py.__file__)


def write_to_h5_file(
        data_files,
        label_files,
        tfname,
        start_entry,
        all_data_shape,
        all_label_shape,
        tell_progress=10,
):
    if MPI.COMM_WORLD.rank == 0 and os.path.isfile(tfname):
        print("file exists")
        exit(1)
        os.remove(tfname)
    # MPI.COMM_WORLD.Barrier()
    print("creating file")
    with h5py.File(
            tfname,
            'w',
            driver='mpio',
            comm=MPI.COMM_WORLD,
            libver='latest',
    ) as fi:
        print("creating dset")
        dset = fi.create_dataset('data', all_data_shape, dtype='f')
        print("creating lset")
        lset = fi.create_dataset('labels', all_label_shape, dtype='f')

        startt = time.time()
        for ii, (f, l) in enumerate(zip(data_files, label_files)):
            data = np.load(f)
            label = np.load(l)
            dset[start_entry + ii] = data
            lset[start_entry + ii] = label
            now = time.time()
            time_remaining = len(data_files) * (now - startt) / (ii + 1)
            if ii % tell_progress == 0:
                print(ii, time_remaining / 60, f, l)


if __name__ == "__main__":
    for data_subset in ["train", "validation"]:
        print(data_subset)

        hdf5_file_name = os.path.join(out_dir, f"{data_subset}.h5")
        files_file_name = os.path.join(out_dir, f"{data_subset}.h5.files")

        target_dir = os.path.join(root_dir, data_subset)
        target_files = os.listdir(target_dir)

        data_files = list(filter(
            lambda x: x.endswith("npy") and x.startswith("data"),
            target_files,
        ))
        data_files.sort()
        data_files = np.array(data_files)
        label_files = list(filter(
            lambda x: x.endswith("npy") and x.startswith("label"),
            target_files,
        ))
        label_files.sort()
        label_files = np.array(label_files)

        if using_cscratch and MPI.COMM_WORLD.rank == 0:
            # Initialize IME cache.
            for data_file in data_files:
                os.system(
                    'ime-ctl --prestage '
                    + os.path.join(target_dir, data_file)
                )
            for label_file in label_files:
                os.system(
                    'ime-ctl --prestage '
                    + os.path.join(target_dir, label_file)
                )
        # Make sure IME cache is initialized before continuing.
        MPI.COMM_WORLD.Barrier()

        perm = np.random.permutation(len(label_files))
        data_files = data_files[perm]
        label_files = label_files[perm]

        no_shards = MPI.COMM_WORLD.size
        data_files_filtered = data_files[:]
        label_files_filtered = label_files[:]
        data_files_shards = []
        label_files_shards = []
        for i in range(no_shards):
            shard_size = int(np.ceil(len(data_files_filtered) / no_shards))
            start = i * shard_size
            end = min((i + 1) * shard_size, len(data_files_filtered))
            data_files_shards.append(data_files_filtered[start:end])
            label_files_shards.append(label_files_filtered[start:end])

        start_entries = np.cumsum([len(x) for x in data_files_shards])
        start_entries = ([0] + list(start_entries))[:-1]

        data_shape = np.load(os.path.join(target_dir, data_files[0])).shape
        label_shape = np.load(os.path.join(target_dir, label_files[0])).shape
        all_data_shape = (len(data_files_filtered), *data_shape)
        all_label_shape = (len(data_files_filtered), *label_shape)

        rank = MPI.COMM_WORLD.rank
        my_data_files = [
            os.path.join(target_dir, f) for f in data_files_shards[rank]
        ]
        my_label_files = [
            os.path.join(target_dir, f) for f in label_files_shards[rank]
        ]
        start_entry = start_entries[rank]

        all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(
            my_data_files))

        print(len(np.unique(all_data_files)), "unique files and",
              len(all_data_files), "total files.")
        if len(np.unique(all_data_files)) != len(all_data_files):
            print("There is an error with the file distribution")

        with open(files_file_name, "w") as files_file:
            files_file.write("\n".join(all_data_files))

        write_to_h5_file(
            [os.path.join(target_dir, f) for f in my_data_files],
            [os.path.join(target_dir, f) for f in my_label_files],
            hdf5_file_name,
            start_entry,
            all_data_shape,
            all_label_shape,
        )
        MPI.COMM_WORLD.Barrier()
        if using_cscratch and MPI.COMM_WORLD.rank == 0:
            # Synchronize file system with IME cache.
            os.system('ime-ctl --sync ' + hdf5_file_name)
            os.system('ime-ctl --sync ' + files_file_name)
        MPI.COMM_WORLD.Barrier()

    print('done')
