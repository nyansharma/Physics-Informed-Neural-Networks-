'''
This script is used to make histograms of any features from the clouds_condensed h5 files.
'''
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor, as_completed

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
from pathlib import Path
import re
import ast
import os
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system') 

# Define the base directory
base_dir = Path('/ceph/cms/store/group/mds-ml/clouds_condensed')
fileset = list(base_dir.rglob('*.h5'))
sig_fileset = [f for f in fileset if f.name.startswith("MS")]
bkg_fileset = [f for f in fileset if not f.name.startswith("MS")]

# Define variables you're interested in plotting
vars = ['cscRechitClusterSkewX', 'cscRechitClusterSkewY']

num_vars = len(vars)

# Function to process a background file
def process_bkg_file(file, file_idx):
    QCD_values = [[] for _ in range(num_vars)]
    Gun_values = [[] for _ in range(num_vars)]
    bkg_values = [[] for _ in range(num_vars)]
    
    with h5py.File(file, 'r') as h5file:
        labels = h5file['label'][:]
        idx_QCD = np.where(labels == 1)[0]
        idx_Gun = np.where(labels == 2)[0]

        for j, var in enumerate(vars):
            dataset = h5file[var][:]

            # Use list comprehension to quickly grab the slices
            QCD_values[j] = dataset[idx_QCD]
            Gun_values[j] = dataset[idx_Gun]
    
    QCD_values = [np.array(v) for v in QCD_values]
    Gun_values = [np.array(v) for v in Gun_values]
    bkg_values = [np.concatenate([QCD_values[i], Gun_values[i]]) for i in range(num_vars)]
    
    return QCD_values, Gun_values, bkg_values


# Function to process a signal file
def process_sig_file(file, file_idx):
    ggH_values = [[] for _ in range(num_vars)]
    
    with h5py.File(file, 'r') as h5file:        
        for j, var in enumerate(vars):
            dataset = h5file[var][:]
            ggH_values[j] = dataset

    ggH_values = [np.array(v) for v in ggH_values]
    
    return ggH_values

# Create empty arrays to store all the feature data for each type
ggH_data = [[] for _ in range(num_vars)]
QCD_data = [[] for _ in range(num_vars)]
Gun_data = [[] for _ in range(num_vars)]
bkg_data = [[] for _ in range(num_vars)] # combines QCD and Gun

# Loads signal data in parallel
with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers as needed
    try:
        futures = {executor.submit(process_sig_file, file, str(file).split('_')[-1].split('.')[0]): file for file in sig_fileset}
    
        for future in as_completed(futures):
            ggH = future.result()
            for i in range(num_vars):
                ggH_data[i].extend(ggH[i])
    
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
        executor.shutdown(wait=True, cancel_futures=True)
        exit(1)

# Loads background data in parallel
with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers as needed
    try:
        futures = {executor.submit(process_bkg_file, file, str(file).split('_')[-1].split('.')[0]): file for file in bkg_fileset}
    
        for future in as_completed(futures):
            QCD, Gun, bkg = future.result()
            for i in range(num_vars):
                QCD_data[i].extend(QCD[i])
                Gun_data[i].extend(Gun[i])
                bkg_data[i].extend(bkg[i])

    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Shutting down...")
        executor.shutdown(wait=True, cancel_futures=True)
        exit(1)


save_path = "/home/users/expan/mds-ml/images/universal_refs/histograms/csc/clouds/cluster_level"
for i, var in enumerate(vars):
    ggH_data_flat = np.array([x.item() for x in ggH_data[i]])
    QCD_data_flat = np.array([x.item() for x in QCD_data[i]])
    Gun_data_flat = np.array([x.item() for x in Gun_data[i]])
    bkg_data_flat = np.array([x.item() for x in bkg_data[i]])
    
    ggH_avg = np.mean(ggH_data_flat)
    QCD_avg = np.mean(QCD_data_flat)
    Gun_avg = np.mean(Gun_data_flat)
    bkg_avg = np.mean(bkg_data_flat)
    
    # Plot the histograms
    plt.figure()
    plt.hist(ggH_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean ggH = {ggH_avg:.2f}") 
    plt.hist(QCD_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean QCD = {QCD_avg:.2f}")
    plt.hist(Gun_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean Gun = {Gun_avg:.2f}")
    plt.title(f"{var}")
    plt.legend()
    plt.savefig(f"{save_path}/{var}.png")
    plt.close()
    
    plt.figure()
    plt.hist(ggH_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean ggH = {ggH_avg:.2f}") 
    plt.hist(bkg_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean background = {bkg_avg:.2f}") 
    plt.title(f"{var}")
    plt.legend()
    plt.savefig(f"{save_path}/{var}_sig_bkg.png")
    plt.close()

    plt.figure()
    plt.hist(ggH_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean ggH = {ggH_avg:.2f}") 
    plt.hist(QCD_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean QCD = {QCD_avg:.2f}")
    plt.hist(Gun_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean Gun = {Gun_avg:.2f}")
    plt.hist(bkg_data_flat, bins="rice", log=True, density=True, histtype="step", label=f"mean background = {bkg_avg:.2f}")
    plt.title(f"{var}")
    plt.legend()
    plt.savefig(f"{save_path}/{var}_all.png")
    plt.close()



