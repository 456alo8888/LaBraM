import h5py
import numpy as np
from pathlib import Path

import torch
from typing import List 
import pickle 
import lmdb 
from torch.utils.data import Dataset 
from eegUtils import to_tensor

class h5Dataset:
    def __init__(self, path:Path, name:str) -> None:
        self.__name = name
        self.__f = h5py.File(path / f'{name}.hdf5', 'a')
    
    def addGroup(self, grpName:str):
        return self.__f.create_group(grpName)
    
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, chunks:tuple):
        return grp.create_dataset(dsName, data=arr, chunks=chunks)
    
    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue
    
    def save(self):
        self.__f.close()
    
    @property
    def name(self):
        return self.__name



class MergedPretrainingDataset(Dataset):
    def __init__(
            self,
            dataset_dirs: List[str]
    ):
        super(MergedPretrainingDataset, self).__init__()
        self.dbs = [lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False) for dataset_dir in dataset_dirs]
        self.keys = []
        for db_idx, db in enumerate(self.dbs):
            with db.begin(write=False) as txn:
                raw = txn.get('__keys__'.encode())
                if raw is None:
                    continue
                keys = pickle.loads(raw)
                # Store (key, db_idx) pairs
                self.keys.extend((k, db_idx) for k in keys)
        # self.keys = self.keys[:100000]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key, db_idx = self.keys[idx]

        with self.dbs[db_idx].begin(write=False) as txn:
            key_bytes = key if isinstance(key, bytes) else key.encode()
            raw = txn.get(key_bytes)
            if raw is None:
                raise KeyError(f"Key '{key}' not found in LMDB index {db_idx}")
            patch = pickle.loads(raw)

        patch = to_tensor(patch)
        # print(patch.shape)
        return patch