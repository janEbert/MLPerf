import io
import os
import time

import h5py
from mpi4py import MPI
import numpy as np
import tarfile


def load_numpy(tar_file, tar_member):
    with tar_file.extractfile(tar_member) as f:
        return np.load(io.BytesIO(f.read()))


# We do _not_ want to preshuffle! CosmoFlow is very sensitive to data
# order/distribution.
preshuffle = False
seed = 42

project_name = 'hai_mlperf'
# Try to use CSCRATCH, fall back to SCRATCH.
using_cscratch = ('CSCRATCH_' + project_name) in os.environ
project_dir = os.getenv(
    'CSCRATCH_' + project_name,
    os.getenv('SCRATCH_' + project_name),
)

# Read file
tar_file_name = os.path.join(
    project_dir, "cosmoUniverse_2019_05_4parE_tf_v2_numpy.tar")
out_dir = os.path.join(project_dir, "cosmoflow")
if MPI.COMM_WORLD.rank == 0:
    os.makedirs(out_dir, exist_ok=True)

np.random.seed(seed)

print("this is the h5py file we are using:", h5py.__file__)

for data_subset in ['train', 'validation']:
    # Write files
    hdf5_file_name = os.path.join(out_dir, f"{data_subset}.h5")
    files_file_name = os.path.join(out_dir, f"{data_subset}.h5.files")

    if using_cscratch and MPI.COMM_WORLD.rank == 0:
        # Initialize IME cache.
        os.system('ime-ctl --prestage ' + tar_file_name)
        # os.system('ime-ctl --prestage ' + hdf5_file_name)
        # os.system('ime-ctl --prestage ' + files_file_name)
    # Make sure IME cache is initialized before continuing.
    MPI.COMM_WORLD.Barrier()

    with tarfile.open(tar_file_name, 'r') as tar_f:
        start_time = time.perf_counter()
        files = [
            n
            for n in tar_f.getmembers()
            if n.name.startswith(
                    f'cosmoUniverse_2019_05_4parE_tf_v2_numpy/{data_subset}')
        ]
        print(f'{MPI.COMM_WORLD.rank}: reading members took',
              time.perf_counter() - start_time, 'seconds')

        data_files = list(filter(lambda x: x.name.endswith("data.npy"), files))
        data_files.sort(key=lambda x: x.name)
        data_files = np.array(data_files)
        label_files = list(filter(
            lambda x: x.name.endswith("label.npy"), files))
        label_files.sort(key=lambda x: x.name)
        label_files = np.array(label_files)
        if preshuffle:
            perm = np.random.permutation(len(label_files))
            data_files = data_files[perm]
            label_files = label_files[perm]

        no_shards = MPI.COMM_WORLD.size
        data_files_filtered = data_files[:]
        label_files_filtered = label_files[:]
        data_files_shards = []
        label_files_shards = []
        for i in range(no_shards):
            shard_size = int(np.ceil(len(data_files_filtered)/no_shards))
            start = i * shard_size
            end = min((i + 1) * shard_size, len(data_files_filtered))
            data_files_shards.append(data_files_filtered[start:end])
            label_files_shards.append(label_files_filtered[start:end])

        start_entries = np.cumsum([len(x) for x in data_files_shards])
        start_entries = ([0] + list(start_entries))[:-1]

        first_data = load_numpy(tar_f, data_files[0])
        data_shape = first_data.shape
        data_dtype = first_data.dtype
        first_label = load_numpy(tar_f, label_files[0])
        label_shape = first_label.shape
        label_dtype = first_label.dtype
        all_data_shape = (len(data_files_filtered),) + data_shape
        all_label_shape = (len(data_files_filtered),) + label_shape

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
            MPI.COMM_WORLD.Barrier()
            print("creating file")
            with h5py.File(
                    tfname,
                    'w',
                    driver='mpio',
                    comm=MPI.COMM_WORLD,
                    libver='latest',
            ) as fi:
                print("creating dset")
                dset = fi.create_dataset(
                    'data', all_data_shape, dtype=data_dtype)
                print("creating lset")
                lset = fi.create_dataset(
                    'label', all_label_shape, dtype=label_dtype)

                start = time.time()
                for i, (f, l) in enumerate(zip(data_files, label_files)):
                    data = load_numpy(tar_f, f)
                    label = load_numpy(tar_f, l)
                    dset[start_entry + i] = data
                    lset[start_entry + i] = label
                    now = time.time()
                    time_remaining = len(data_files) * (now-start) / (i + 1)
                    if i % tell_progress == 0:
                        print(i, time_remaining/60, f.name, l.name)
                fi.close()

        my_data_files = [f for f in data_files_shards[MPI.COMM_WORLD.rank]]
        my_label_files = [f for f in label_files_shards[MPI.COMM_WORLD.rank]]
        start_entry = start_entries[MPI.COMM_WORLD.rank]

        all_data_files = np.concatenate(MPI.COMM_WORLD.allgather(
            [m.name for m in my_data_files]))

        print(
            len(np.unique(all_data_files)),
            "unique files, and",
            len(all_data_files),
            "total files.",
        )
        if len(np.unique(all_data_files)) != len(all_data_files):
            print("There is an error with the file distribution")

        if MPI.COMM_WORLD.rank == 0:
            with open(files_file_name, "w") as g:
                g.write("\n".join(all_data_files) + '\n')

        write_start_time = time.perf_counter()
        write_to_h5_file(
            [f for f in my_data_files],
            [f for f in my_label_files],
            hdf5_file_name,
            start_entry,
            all_data_shape,
            all_label_shape,
        )

        print('finished', data_subset, 'on', MPI.COMM_WORLD.rank, 'after',
              time.perf_counter() - write_start_time, 'seconds')

        MPI.COMM_WORLD.Barrier()
        if using_cscratch and MPI.COMM_WORLD.rank == 0:
            # Synchronize file system with IME cache.
            os.system('ime-ctl --sync ' + hdf5_file_name)
            os.system('ime-ctl --sync ' + files_file_name)
        MPI.COMM_WORLD.Barrier()
