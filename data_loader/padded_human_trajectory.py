import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PaddedHumanTrajectoryDataset(Dataset):
    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002, min_ped=1, delim='\t',
                 augment=False, augment_angle=15.0):
        super().__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = obs_len + pred_len
        self.delim = delim
        self.threshold = threshold
        self.min_ped = min_ped
        self.augment = augment
        self.augment_angle = augment_angle

        all_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        all_sequences = []
        max_people = 0
        
        for path in all_files:
            data = self._read_file(path, self.delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == frame, :] for frame in frames]
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
            
            for idx in range(0, num_sequences * skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                _non_linear_ped = []
                num_peds_considered = 0
                
                for ped_idx, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    _non_linear_ped.append(self._poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                
                if num_peds_considered > min_ped:
                    all_sequences.append({
                        'seq': curr_seq[:num_peds_considered],
                        'seq_rel': curr_seq_rel[:num_peds_considered],
                        'loss_mask': curr_loss_mask[:num_peds_considered],
                        'non_linear_ped': _non_linear_ped,
                        'num_peds': num_peds_considered
                    })
                    max_people = max(max_people, num_peds_considered)
        
        self.max_people = max_people
        self.num_seq = len(all_sequences)
        
        if self.num_seq == 0:
            raise RuntimeError(f"No valid sequences found in {data_dir}")
        
        print(f"Dataset: {data_dir}")
        print(f"Total sequences: {self.num_seq}")
        print(f"Max people per sequence: {self.max_people}")
        
        self.sequences = []
        for seq_data in all_sequences:
            padded_seq = self._pad_sequence(seq_data)
            self.sequences.append(padded_seq)

    def _pad_sequence(self, seq_data):
        seq = seq_data['seq']
        seq_rel = seq_data['seq_rel']
        loss_mask = seq_data['loss_mask']
        non_linear_ped = seq_data['non_linear_ped']
        num_peds = seq_data['num_peds']
        
        padded_seq = np.zeros((self.max_people, 2, self.seq_len))
        padded_seq_rel = np.zeros((self.max_people, 2, self.seq_len))
        padded_loss_mask = np.zeros((self.max_people, self.seq_len))
        padded_non_linear_ped = np.zeros(self.max_people)
        
        padded_seq[:num_peds] = seq
        padded_seq_rel[:num_peds] = seq_rel
        padded_loss_mask[:num_peds] = loss_mask
        padded_non_linear_ped[:num_peds] = non_linear_ped
        
        validity_mask = np.zeros(self.max_people)
        validity_mask[:num_peds] = 1
        
        return {
            'obs_traj': torch.from_numpy(padded_seq[:, :, :self.obs_len]).float(),
            'pred_traj': torch.from_numpy(padded_seq[:, :, self.obs_len:]).float(),
            'obs_traj_rel': torch.from_numpy(padded_seq_rel[:, :, :self.obs_len]).float(),
            'pred_traj_rel': torch.from_numpy(padded_seq_rel[:, :, self.obs_len:]).float(),
            'loss_mask': torch.from_numpy(padded_loss_mask).float(),
            'non_linear_ped': torch.from_numpy(padded_non_linear_ped).float(),
            'validity_mask': torch.from_numpy(validity_mask).float(),
            'num_peds': num_peds
        }

    def _repad_all_sequences(self):
        new_sequences = []
        for seq in self.sequences:
            validity_mask = seq['validity_mask'].numpy()
            num_valid = int(validity_mask.sum())
            
            obs_traj = seq['obs_traj'].numpy()[:num_valid]
            pred_traj = seq['pred_traj'].numpy()[:num_valid]
            
            full_traj = np.concatenate([obs_traj, pred_traj], axis=2)
            
            padded_seq = np.zeros((self.max_people, 2, self.seq_len))
            padded_seq[:num_valid] = full_traj
            
            new_validity_mask = np.zeros(self.max_people)
            new_validity_mask[:num_valid] = 1
            
            new_seq = {
                'obs_traj': torch.from_numpy(padded_seq[:, :, :self.obs_len]).float(),
                'pred_traj': torch.from_numpy(padded_seq[:, :, self.obs_len:]).float(),
                'obs_traj_rel': seq['obs_traj_rel'],
                'pred_traj_rel': seq['pred_traj_rel'],
                'loss_mask': seq['loss_mask'],
                'non_linear_ped': seq['non_linear_ped'],
                'validity_mask': torch.from_numpy(new_validity_mask).float(),
                'num_peds': seq['num_peds']
            }
            new_sequences.append(new_seq)
        
        self.sequences = new_sequences

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        seq = self.sequences[index]
        
        if self.augment and self.augment_angle > 0:
            angle_deg = np.random.uniform(-self.augment_angle, self.augment_angle)
            angle_rad = np.deg2rad(angle_deg)
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a],
                                       [sin_a, cos_a]], dtype=np.float32)
            
            obs_traj = seq['obs_traj'].numpy()
            pred_traj = seq['pred_traj'].numpy()
            
            obs_traj_T = obs_traj.transpose(0, 2, 1)
            pred_traj_T = pred_traj.transpose(0, 2, 1)
            
            obs_traj_rot = obs_traj_T @ rotation_matrix.T
            pred_traj_rot = pred_traj_T @ rotation_matrix.T
            
            obs_traj_rot = obs_traj_rot.transpose(0, 2, 1)
            pred_traj_rot = pred_traj_rot.transpose(0, 2, 1)
            
            seq = {
                'obs_traj': torch.from_numpy(obs_traj_rot).float(),
                'pred_traj': torch.from_numpy(pred_traj_rot).float(),
                'obs_traj_rel': seq['obs_traj_rel'],
                'pred_traj_rel': seq['pred_traj_rel'],
                'loss_mask': seq['loss_mask'],
                'non_linear_ped': seq['non_linear_ped'],
                'validity_mask': seq['validity_mask'],
                'num_peds': seq['num_peds']
            }
        
        return seq

    @staticmethod
    def _read_file(_path, delim='\t'):
        data = []
        if delim == 'tab':
            delim = '\t'
        elif delim == 'space':
            delim = ' '
        with open(_path, 'r') as f:
            for line in f:
                line = line.strip().split(delim)
                line = [float(i) for i in line]
                data.append(line)
        return np.asarray(data)

    @staticmethod
    def _poly_fit(traj, traj_len, threshold):
        t = np.linspace(0, traj_len - 1, traj_len)
        res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
        res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
        if res_x + res_y >= threshold:
            return 1.0
        else:
            return 0.0


def get_padded_human_trajectory_dataloader(data_dir, batch_size=64, obs_len=8, pred_len=12, 
                                          skip=1, threshold=0.002, min_ped=1, delim='\t',
                                          shuffle=True, num_workers=4, augment=False, augment_angle=15.0):
    dataset = PaddedHumanTrajectoryDataset(
        data_dir, obs_len=obs_len, pred_len=pred_len, skip=skip,
        threshold=threshold, min_ped=min_ped, delim=delim,
        augment=augment, augment_angle=augment_angle
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataset, dataloader


def get_padded_human_trajectory_dataloaders(cfg):
    data_cfg = cfg.get('data', {})
    root = data_cfg.get('root', './data/human_trajectory')
    batch_size = int(data_cfg.get('batch_size', 32))
    num_workers = int(data_cfg.get('num_workers', 4))
    obs_len = int(data_cfg.get('obs_len', 8))
    pred_len = int(data_cfg.get('pred_len', 12))
    skip = int(data_cfg.get('skip', 1))
    threshold = float(data_cfg.get('threshold', 0.002))
    min_ped = int(data_cfg.get('min_ped', 1))
    delim = data_cfg.get('delim', '\t')
    
    augment = bool(data_cfg.get('augment', False))
    augment_angle = float(data_cfg.get('augment_angle', 15.0))
    
    dataset_name = data_cfg.get('dataset', 'eth')
    
    train_path = os.path.join(root, dataset_name, 'train')
    val_path = os.path.join(root, dataset_name, 'val') 
    test_path = os.path.join(root, dataset_name, 'test')
    
    train_dataset, train_loader = get_padded_human_trajectory_dataloader(
        train_path, batch_size=batch_size, obs_len=obs_len, pred_len=pred_len,
        skip=skip, threshold=threshold, min_ped=min_ped, delim=delim,
        shuffle=True, num_workers=num_workers,
        augment=augment, augment_angle=augment_angle
    )
    
    val_dataset, val_loader = get_padded_human_trajectory_dataloader(
        val_path, batch_size=batch_size, obs_len=obs_len, pred_len=pred_len,
        skip=skip, threshold=threshold, min_ped=min_ped, delim=delim,
        shuffle=False, num_workers=num_workers,
        augment=False, augment_angle=0.0
    )
    
    test_dataset, test_loader = get_padded_human_trajectory_dataloader(
        test_path, batch_size=batch_size, obs_len=obs_len, pred_len=pred_len,
        skip=skip, threshold=threshold, min_ped=min_ped, delim=delim,
        shuffle=False, num_workers=num_workers,
        augment=False, augment_angle=0.0
    )
    
    global_max_people = max(train_dataset.max_people, val_dataset.max_people, test_dataset.max_people)
    
    if train_dataset.max_people != global_max_people:
        train_dataset.max_people = global_max_people
        train_dataset._repad_all_sequences()
    if val_dataset.max_people != global_max_people:
        val_dataset.max_people = global_max_people
        val_dataset._repad_all_sequences()
    if test_dataset.max_people != global_max_people:
        test_dataset.max_people = global_max_people
        test_dataset._repad_all_sequences()
    
    print(f"Human Trajectory Dataset: {dataset_name.upper()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Obs length: {obs_len}, Pred length: {pred_len}")
    print(f"Max people per sequence (global): {global_max_people}")
    
    return train_loader, val_loader, test_loader