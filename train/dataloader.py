import torch
from torch.utils.data import Dataset
import numpy as np

class AudioSet(Dataset):
    def __init__(self, data, arousal_fusion=True):
        # 'np.nan_to_num' replaces NaN with zero and infinity with large finite numbers (default behaviour)
        
        self.arousal_fusion = arousal_fusion  # arousal audio and video fusion
        self.arousal_audio_feature = torch.Tensor(np.nan_to_num(np.array(data["arousal_audio_feature"])))
        self.valence_audio_feature = torch.Tensor(np.nan_to_num(np.array(data["valence_audio_feature"])))
        self.arousal_video_feature = torch.Tensor(np.nan_to_num(np.array(data["arousal_video_feature"])))
        self.valence_video_feature = torch.Tensor(np.nan_to_num(np.array(data["valence_video_feature"])))
        self.arousal_label = torch.Tensor(np.array(data["arousal_label"]))
        self.valence_label = torch.Tensor(np.array(data["valence_label"]))

    def __len__(self):
        return len(self.arousal_audio_feature)

    def __getitem__(self, index):
        if self.arousal_fusion:
            arousal_audio_feature = self.arousal_audio_feature[index]
            arousal_video_feature = self.arousal_video_feature[index]
            arousal_label = self.arousal_label[index]
            return arousal_audio_feature, arousal_video_feature, arousal_label
        else:
            valence_audio_feature = self.valence_audio_feature[index]
            valence_video_feature = self.valence_audio_feature[index]
            valence_label = self.valence_label[index]
            return valence_audio_feature, valence_video_feature, valence_label