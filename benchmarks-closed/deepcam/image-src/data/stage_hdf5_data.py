import os
import mmap
from glob import glob
import itertools
import numpy as np
import argparse as ap
import concurrent.futures as cf
import time
from queue import LifoQueue as Queue
import torch.cuda.nvtx as nvtx
import copy

import torch
import h5py
import subprocess


# small sharding helper function
def get_shard_ranges(num_files, num_shards, shard_id, cycle_dist=0):
    num_files = len(files)
    dummy_

    # shard files into bulk and remainder:
    num_files_per_shard = num_files // num_shards
    files_bulk = files[:num_files_per_shard * num_shards]
    files_remainder = files[num_files_per_shard * num_shards:]
    
    # get my shard
    files_shard = []
    shard_start = shard_id * num_files_per_shard
    shard_end = shard_start + num_files_per_shard
    files_shard.append(range(shard_start,shard_end))
    #files_shard = files_bulk[shard_start:shard_end].copy()
    
    # deal with remainder: round robin with some offset for even distribution
    cycle_offset = 0
    for idf, fname in enumerate(files_remainder):
        if ((idf + cycle_offset) % num_shards == shard_id):
            files_shard.append(fname)
        cycle_offset += cycle_dist
            
    return files_shard  

def get_shard_range(num_files, num_shards, shard_id, cycle_dist=0):
    assert(shard_id<num_shards)
    # shard files into bulk and remainder:
    num_files_per_shard = num_files // num_shards
    num_files_bulk = num_files_per_shard * num_shards
    num_files_remainder = num_files%num_shards
    # We drop all files above the remainder
    print("num_files_remainder", num_files_remainder)

    shard_start=[0]
    for i in range(1,num_shards):
        if i-1 < num_files_remainder:
            this_shard_start=shard_start[-1]+(num_files_per_shard+1)
        else:
            this_shard_start=shard_start[-1]+(num_files_per_shard)
        shard_start.append(this_shard_start)
    shard_start.append(num_files)

    ranges=[]
    for i in range(num_shards):
        ranges.append((shard_start[i], shard_start[i+1]))
    return ranges[shard_id]


def finalize_load_local(fdata_handles):
    nvtx.range_push("finalize_load_local")
    file_data = []
    rbytes = 0
    while not fdata_handles.empty():
        handle = fdata_handles.get()
        if handle.done():
            fname, fdata, fbytes = handle.result()
            file_data.append((fname, fdata))
            rbytes += fbytes
        else:
            fdata_handles.put(handle)
    nvtx.range_pop()
    
    return file_data, rbytes


def finalize_load_global(comm, files, rbytes):
    nvtx.range_push("finalize_load_global")
    files_all, rbytes = exchange_buffers(comm, files)
    #we can compute that from the gathered data, no need to sync up
    #rbytes = comm.allreduce(rbytes)
    nvtx.range_pop()
    
    return files_all, rbytes


def load_file(filename):
    with open(filename, "rb") as f:
        token = f.read()

    return filename, token, len(token)


def load_file_direct(filename, filesize=None):
    if False:
        # open file
        fd = os.open(filename, os.O_RDONLY | os.O_DIRECT)
        
        # check file size:
        if filesize is None:
            stat = os.fstat(fd)
            readsize = stat.st_size
        else:
            readsize = filesize
        
        # create mmap
        mm = mmap.mmap(fd, readsize, offset=0, access=mmap.ACCESS_READ)
        token = mm.read(readsize)
        mm.close()
        
        # close file
        os.close(fd)
    else:
        blocksize = 4096
        token = ioh.load_file_direct(filename, blocksize=blocksize, filesize=0 if filesize is None else filesize)
        
    return filename, token, len(token)


def load_batch_parallel(executor, files, comm_size, comm_rank, filesize=None, direct_io=False):
    nvtx.range_push("load_batch_parallel")
    # split the data across ranks
    if comm_size > 1:
        files_load = get_shard(files, comm_size, comm_rank, cycle_dist=0)
    else:
        files_load = files

    # submit loads:
    queue = Queue()
    for filename in files_load:
        if direct_io:
            queue.put(executor.submit(load_file_direct, filename, filesize))
        else:
            queue.put(executor.submit(load_file, filename))
    nvtx.range_pop()
    
    return queue


def load_batch(files, comm_size, comm_rank):
    nvtx.range_push("load_batch")
    # split the data across ranks
    if comm_size > 1:
        files_load = get_shard(files, comm_size, comm_rank, cycle_dist=0)
    else:
        files_load = files

    # keep track of how many bytes we load
    rbytes = 0
    result = []
    for fname in files_load:
        _, token, size = load_file(fname)
        result.append((fname, token))
        rbytes += size
    nvtx.range_pop()

    return result, rbytes


def finalize_save_local(fdata_handles):
    nvtx.range_push("finalize_save_local")
    wbytes = 0
    while not fdata_handles.empty():
        handle = fdata_handles.get()
        if handle.done():
            fbytes = handle.result()
            wbytes += fbytes
        else:
            fdata_handles.put(handle)
    nvtx.range_pop()

    return wbytes 
        

def save_file(ofname, fdata):
    with open(ofname, "wb") as f:
        f.write(fdata)
    return len(fdata)


def save_batch_parallel(executor, output_dir, fdata):
    nvtx.range_push("save_batch_parallel")
    queue = Queue()
    for fn, fd in fdata:
        ofn = os.path.join(output_dir, os.path.basename(fn))
        queue.put(executor.submit(save_file, ofn, copy.copy(fd)))
    nvtx.range_pop()

    return queue


def save_batch(output_dir, fdata):
    nvtx.range_push("save_batch")
    wbytes = 0.
    for fn, fd in fdata:
        ofn = os.path.join(output_dir, os.path.basename(fn))
        wbytes += save_file(ofn, fd)
    nvtx.range_pop()
    
    return wbytes


def exchange_buffers(comm, fdata):
    # quick exit if we do not need to communicate
    if comm.Get_size() == 1:
        rbytes = sum([len(x[1]) for x in fdata])
        return fdata, rbytes

    # profile region start
    nvtx.range_push("exchange_buffers")

    # gather
    fdata_all = comm.allgather(fdata)
    
    # flatten
    fdata_result = list(itertools.chain(*fdata_all))
        
    # size:
    rbytes = sum([len(x[1]) for x in fdata_result])

    # stop profiling
    nvtx.range_pop()

    # return
    return fdata_result, rbytes


# this routine stages data for each instance
def stage_instance_data(stage_comm, instance_comm, instance_node_comm,
                        lsize, lrank,
                        hdf5file, dataset, target_directory,
                        batch_size=-1,
                        stage_num_workers=1,
                        stage_mode="node",
                        full_dataset_per_node=True,
                        use_direct_io=False,
                        prepare_staging=False, load_hdf5=False):

    # comm parameters
    ssize = stage_comm.Get_size()
    srank = stage_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    nsize = instance_node_comm.Get_size()
    nrank = instance_node_comm.Get_rank()
    print(hdf5file)
    f = h5py.File(hdf5file, "r")
    print("getting dataset", dataset)
    ds = f.get(dataset)
    num_files = ds.shape[0]
#    num_files = 100 # limited for testing

    shard_start, shard_end = get_shard_range(num_files, isize, irank, cycle_dist=lsize)
    print("shart_start", shard_start, " on rank ", irank)

    chunk_size=16
    try:
        tag=os.path.basename(hdf5file).split(".")[0]
    except Exception as ex:
        print(ex)
        tag="123"
    print(tag)
    chunk_start=shard_start
    files_local=[]
    while True:
         print("this chunk starts at", chunk_start, "on", irank)
         chunk_end=min(shard_end, chunk_start+chunk_size)
         data=ds[chunk_start:chunk_end]
         for i in range(data.shape[0]):
             if dataset=="labels":
                 id_="label"
             else:
                 id_=dataset
             outputfile=id_+"-" + "{:06}".format(chunk_start+i) + ".npy"
             np.save(os.path.join(target_directory, outputfile), data[i])
             files_local.append(outputfile)
         #with open(os.path.join(target_directory, tag + "_" + dataset + ".lst"), "w") as f:
         #    f.write("\n".join(files_local))
         if chunk_end==shard_end:
             break
         chunk_start=chunk_end 
    
    instance_comm.Barrier()
    return 0,0

         

    if True:    
        if full_dataset_per_node:
            files_print = files
        else:
            local_comm = instance_comm.Split(color=(irank // lsize), key=lrank)
            files_print = local_comm.allgather(files_shard)
            files_print = list(itertools.chain(*files_print))
        
        # create tags
        if lrank == 0:
            with open(fname, "w") as f:
                f.write("\n".join(files_print))
        return 0, 0

    else:
        # load metadata
        with open(fname, "r") as f:
            files_shard = f.read().splitlines()

        # shard locally
        files_shard = get_shard(files_shard, lsize, lrank)
        
        
    # now, let's take care of the data: update the number of files because of remainder
    num_files_per_shard = len(files_shard)
    if batch_size > 0:
        num_batches = (num_files_per_shard + batch_size - 1) // batch_size
    else:
        num_batches = 1
        batch_size = num_files_per_shard
        
    # create executor
    executor = cf.ThreadPoolExecutor(max_workers = stage_num_workers)

    # get file sizes:
    filesize = os.stat(files_shard[0]).st_size
        
    
    # submit first batch
    #run_times = {"load_batch_parallel": 0.,
    #             "finalize_load_local": 0.,
    #             "finalize_load_global": 0.,
    #             "save_batch_parallel": 0.,
    #             "finalize_save_local": 0.}
    #ns_to_s = 1e-9
    #b_to_gb = 1./float(1024*1024*1024)

    #t_start = time.perf_counter_ns()
    fdata_handles = load_batch_parallel(executor, files_shard[0:min(num_files_per_shard, batch_size)],
                                        ssize if stage_mode == "global" else 1,
                                        srank if stage_mode == "global" else 0,
                                        filesize,
                                        direct_io = use_direct_io)
    #t_stop = time.perf_counter_ns()
    #run_times["load_batch_parallel"] += (t_stop - t_start) * ns_to_s

    # batch loop: go one step further to process the last batch
    save_handles = []
    total_bytes_read = 0
    total_bytes_write = 0
    for bid in range(1, num_batches+1):
        # wait for data loader
        #t_start = time.perf_counter_ns()
        fdata_save, rbytes = finalize_load_local(fdata_handles)
        #t_stop = time.perf_counter_ns()
        #run_times["finalize_load_local"] += (t_stop - t_start) * ns_to_s
        
        # finalize the load if required:
        if stage_mode == "global":
            #t_start = time.perf_counter_ns()
            fdata_save, rbytes = finalize_load_global(stage_comm, fdata_save, rbytes)
            #t_stop = time.perf_counter_ns()
            #run_times["finalize_load_global"] += (t_stop - t_start) * ns_to_s
            
        # increment total bytes read
        total_bytes_read += rbytes
        
        # get next batch, only submit if there are any
        if bid < num_batches:
            batch_start = bid * batch_size
            batch_end = min(batch_start + batch_size, num_files_per_shard)
            #t_start = time.perf_counter_ns()
            fdata_handles = load_batch_parallel(executor, files_shard[batch_start:batch_end],
                                                ssize if stage_mode == "global" else 1,
                                                srank if stage_mode == "global" else 0,
                                                filesize,
                                                direct_io = use_direct_io)
            #t_stop = time.perf_counter_ns()
            #run_times["load_batch_parallel"] += (t_stop - t_start) * ns_to_s

        # exchange buffers inside instance if necessary
        if full_dataset_per_node and (stage_mode != "node"):
            fdata_save, _ = exchange_buffers(instance_node_comm, fdata_save)

        # wait till data is saved
        if save_handles:
            #t_start = time.perf_counter_ns()
            total_bytes_write += finalize_save_local(save_handles)
            #t_stop = time.perf_counter_ns()
            #run_times["finalize_save_local"] += (t_stop - t_start) * ns_to_s
        
        # store locally
        #saved = executor.submit(save_batch, target_directory, fdata_save)
        #t_start = time.perf_counter_ns()
        save_handles = save_batch_parallel(executor, target_directory, fdata_save)
        #t_stop = time.perf_counter_ns()
        #run_times["save_batch_parallel"] += (t_stop - t_start) * ns_to_s
        
    # finalize last batch
    if save_handles:
        #t_start = time.perf_counter_ns()
        total_bytes_write += finalize_save_local(save_handles)
        #t_stop = time.perf_counter_ns()
        #run_times["finalize_save_local"] += (t_stop - t_start) * ns_to_s

    # global barrier
    instance_comm.Barrier()

    # print statistics
    #if True and (irank == 0):
    #    message = f"Instance {srank} timings:\n"
    #    for key in run_times:
    #        message += f"{key} [s]: {run_times[key]:.3f}\n"
    #    # estimate rw bandwidth:
    #    read_time = run_times["load_batch_parallel"] + run_times["finalize_load_local"] + run_times["finalize_load_global"]
    #    message += f"read-bw [GB/s]: {(total_bytes_read * isize * b_to_gb / read_time):.2f}\n"
    #    write_time = run_times["save_batch_parallel"] + run_times["finalize_save_local"]
    #    message += f"write-bw [GB/s]: {(total_bytes_write * isize * b_to_gb / write_time):.2f}\n"
    #    print(message, flush=True)

    return total_bytes_read, total_bytes_write
    

def stage_data_helper(global_comm, num_instances, instance_id, instance_comm,
                      local_size, local_rank, pargs, verify=False, 
                      full_dataset_per_node=True, use_direct_io=False,
                      seed=333,
                      prepare_staging = False):
    # - Every instance needs all the data, so we need inum replicas.
    # - Every rank irank within an instance can stage data_size / isize of the total data
    # - Since there are num_instances ranks working on the same data, we could shard this among
    # those ranks too
    gsize = global_comm.Get_size()
    grank = global_comm.Get_rank()
    isize = instance_comm.Get_size()
    irank = instance_comm.Get_rank()
    lsize = local_size
    lrank = local_rank
    print("gsize", gsize, "grank", grank, "isize", isize, "irank", irank, "lsize", lsize, "lrank", lrank)

    load_hdf5=False
    
    # create staging filter:
    pargs.data_format = "dali-numpy/hdf5" #TODO Fix
    if False and (pargs.data_format == "dali-numpy") or (pargs.data_format == 'dali-es'):
        stage_filter_list = ['validation/data-*.npy', 'validation/label-*.npy',
                             'train/data-*.npy', 'train/label-*.npy']
        print("not hdf5", pargs.data_format)
    elif pargs.data_format == "dali-numpy/hdf5" or True:
        stage_filter_list = [ "train.h5/data", "train.h5/labels", "validation.h5/data", "validation.h5/labels" ]
        load_hdf5=True
        print("hdf5!!")
    elif pargs.data_format == "dali-dummy":
        return
    else:
        raise NotImplementedError(f"Error, data-format {pargs.data_format} not implemented for staging")

    # create subdirectory for each instance, just in case if multiple instances see the same directory
    stage_dir = os.path.join(pargs.stage_dir_prefix, f"instance{instance_id}")
    if lrank == 0:
        os.makedirs(stage_dir, exist_ok = True)
    
    # split the global communicator according to irank: key could be instance_id but we would end up
    # with having all rank 0 on the same node. Now we got:
    # irank = 0: [0, 1, 2, ..... num_instances]
    # irank = 1: [1, 2, 3, ..... 0]]
    #keys = np.roll(np.arange(0, num_instances), irank).tolist()
    #stage_comm = global_comm.Split(color=irank, key=keys[instance_id])
    stage_comm = global_comm.Split(color=irank, key=instance_id)
    
    # split the instance by nodes and create a comm with all matching local ranks by node
    num_nodes_per_instance = isize // lsize
    instance_node_id = irank // lsize
    instance_node_comm = instance_comm.Split(color=lrank, key=instance_node_id)
    
    # stage the statsfile, it is OK to do that beforehand:
    if prepare_staging:
        if grank == 0:
            print("Copying stats.h5", flush=True)
            with open(os.path.join(pargs.data_dir_prefix, "stats.h5"), "rb") as f:
                statsfile = f.read()
        else:
            statsfile = None

        # broadcast the statsfile
        statsfile = global_comm.bcast(statsfile, 0)

        # save it
        if lrank == 0:
            with open(os.path.join(stage_dir, "stats.h5"), "wb") as f:
                f.write(statsfile)

    
    # iterate over staging filters
    file_stats = {}
    for stage_filter in stage_filter_list:

        nvtx.range_push(f"stage {stage_filter}")
        
        if not prepare_staging and (grank == 0):
            print(f"Staging {stage_filter}", flush=True)
        elif grank == 0:
            print(f"Preparing file lists for {stage_filter}", flush=True)

        # get directories
        if not load_hdf5:
            stage_source_directory = os.path.join(pargs.data_dir_prefix, os.path.dirname(stage_filter))
        else:
            tmp=stage_filter.split("/")
            fname, dataset=tmp[0],tmp[1]
            hdf5_file=os.path.join(pargs.data_dir_prefix, fname)
        stage_target_directory = os.path.join(stage_dir, stage_filter.split(".")[0] )

        # create target directory if not exist:
        if local_rank == 0:
            os.makedirs(stage_target_directory, exist_ok = True)
        
        if not load_hdf5:
             # get file info to everybody
             if grank == 0 and not load_hdf5:
                 allfiles = sorted(glob(os.path.join(stage_source_directory, os.path.basename(stage_filter))))
             else:
                 allfiles = None

             # shuffle files if requested
             if (grank == 0) and (not full_dataset_per_node) and (seed is not None):
                 rng = np.random.default_rng(seed)
                 rng.shuffle(allfiles)
                 
             # communicate list of files
             allfiles = global_comm.bcast(allfiles, 0)
        
        # now stage the data so that each rank in each instance has the relevant data
        stage_start = time.perf_counter()
        total_read, total_write = stage_instance_data(stage_comm, instance_comm, instance_node_comm,
                                                      lsize, lrank, 
                                                      hdf5_file, dataset, stage_target_directory,
                                                      pargs.stage_batch_size,
                                                      pargs.stage_num_workers,
                                                      pargs.stage_mode,
                                                      full_dataset_per_node,
                                                      use_direct_io,
                                                      prepare_staging, load_hdf5=load_hdf5)
        stage_stop = time.perf_counter()

        # updating file stats buffer
        file_stats[stage_filter] = 0# len(allfiles) 

        # skip the rest if we want to prep staging only
        if prepare_staging:
            continue
        
        # unit conversion
        unit_convert_gb = 1./float(1024*1024*1024)

        # allreduce:
        total_read = global_comm.allreduce(total_read)
        total_write = global_comm.allreduce(total_write)

        # convert units
        total_read *= unit_convert_gb
        total_write *= unit_convert_gb
        
        # stage duration:
        stage_duration = stage_stop - stage_start
        
        # print
        if grank == 0:
            print(f"""Staging {stage_filter} done.
                      Total number of files: {file_stats[stage_filter]}.
                      Elapsed time {stage_duration:.2f}s. 
                      Read {total_read:.2f} GB (bandwidth: {total_read/stage_duration:.2f} GB/s).
                      Write {total_write:.2f} GB (bandwidth: {total_write/stage_duration:.2f} GB/s).
                   """)

        # verify staging results if requested
        if verify:
            nvtx.range_push(f"stage_verify")
            if local_rank == 0:
                files = glob(os.path.join(stage_target_directory, os.path.basename(stage_filter)))
            else:
                files = []

            if not full_dataset_per_node:
                # if every node hosts a shard, we need to sum the results, if not we need to make sure everybody has the same
                files_full = instance_comm.allgather(files)
                files_full = set(itertools.chain(*files_full))
            else:
                files_full = set(files)
            num_files = len(files_full)

            # strip off the directory
            checkfiles1 = sorted([os.path.basename(x) for x in files_full])
            checkfiles2 = sorted([os.path.basename(x) for x in allfiles])

            assert(num_files == file_stats[stage_filter])
            assert(checkfiles1 == checkfiles2)
                
            if irank == 0:
                print(f"Staged data for {stage_filter}: {num_files}, expected: {file_stats[stage_filter]}", flush=True)
            nvtx.range_pop()

        # close range
        nvtx.range_pop()
            

    #if irank==0:
    #    print("now look at my files")
    #    f=subprocess.check_output(["find", "/tmp/deepcam"])
    #    print(f.decode('utf-8'))
    #    print("/look")
    # make sure we have the right number of files everywhere
    #assert(file_stats['validation/data-*.npy'] == file_stats['validation/label-*.npy'])
    #assert(file_stats['train/data-*.npy'] == file_stats['train/label-*.npy'])
    
    #return file_stats['train/data-*.npy'], file_stats['validation/data-*.npy']
    return 121266, 15158 
