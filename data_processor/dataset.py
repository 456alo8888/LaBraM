import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset


import torch
import pickle 
import lmdb 


list_path = List[Path]

class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
    
    def free(self) -> None: 
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()
    



# class MergedPretrainingDataset(Dataset):
#     def __init__(
#             self,
#             dataset_dirs: List[str]
#     ):
#         super(MergedPretrainingDataset, self).__init__()
#         self.dbs = [lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False) for dataset_dir in dataset_dirs]
#         self.keys = []
#         for db_idx, db in enumerate(self.dbs):
#             with db.begin(write=False) as txn:
#                 raw = txn.get('__keys__'.encode())
#                 if raw is None:
#                     continue
#                 keys = pickle.loads(raw)
#                 # Store (key, db_idx) pairs
#                 self.keys.extend((k, db_idx) for k in keys)
#         # self.keys = self.keys[:100000]

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         key, db_idx = self.keys[idx]

#         with self.dbs[db_idx].begin(write=False) as txn:
#             key_bytes = key if isinstance(key, bytes) else key.encode()
#             raw = txn.get(key_bytes)
#             if raw is None:
#                 raise KeyError(f"Key '{key}' not found in LMDB index {db_idx}")
#             patch = pickle.loads(raw)

#         patch = to_tensor(patch)
#         # print(patch.shape)
#         return patch



def to_tensor(x):
    # bạn có thể thay đổi tùy data
    return torch.tensor(x, dtype=torch.float32)

class MergedPretrainingDataset(Dataset):
    """
    Giống ShockDataset nhưng dữ liệu lưu trong LMDB.
    Mỗi key trong LMDB có thể chứa 1 đoạn dài (vd: 30s), 
    ta sẽ cắt thành nhiều windows ngắn hơn.
    """
    def __init__(self, dataset_dirs, window_size_secs=4, stride_size=200, sample_rate=200, feature_size=None, ch_names=None):
        super().__init__()
        self.dbs = [lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False) 
                    for dataset_dir in dataset_dirs]

        self.window_size = window_size_secs * sample_rate
        self.stride_size = stride_size
        
        # lưu mapping index toàn cục -> (db_idx, key, start_offset)
        self.index_map = []
        for db_idx, db in enumerate(self.dbs):
            with db.begin(write=False) as txn:
                raw = txn.get('__keys__'.encode())
                if raw is None:
                    continue
                keys = pickle.loads(raw)

            for key in keys:
                with db.begin(write=False) as txn:
                    patch = pickle.loads(txn.get(key.encode() if isinstance(key, str) else key))
                length = patch.shape[1]  # số sample theo time
                # số windows có thể sinh ra
                num_windows = (length - self.window_size) // self.stride_size + 1
                for w in range(num_windows):
                    start = w * self.stride_size
                    self.index_map.append((db_idx, key, start))

        self._feature_size = feature_size
        self._ch_names = ch_names

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        db_idx, key, start = self.index_map[idx]
        with self.dbs[db_idx].begin(write=False) as txn:
            raw = txn.get(key if isinstance(key, bytes) else key.encode())
            patch = pickle.loads(raw)

        window = patch[:, start:start+self.window_size]
        window = to_tensor(window)

        if self._feature_size is None:
            self._feature_size = list(window.shape)
        return window

    @property
    def feature_size(self):
        return self._feature_size

    def get_ch_names(self):
        if self._ch_names is None:
            raise ValueError("Channel names (ch_names) không được lưu trong LMDB. "
                             "Bạn cần truyền vào khi tạo dataset.")
        return self._ch_names

    def free(self):
        for db in self.dbs:
            db.close()