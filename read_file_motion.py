import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T

keep_hmdb51 = ["clap", "climb", "drink", "jump", "pour", "ride_bike", "ride_horse",
               "run", "shoot_bow", "smoke", "throw", "wave"]

torch.manual_seed(97)
num_frames = 16
clip_steps = 2
batch_size = 16

transform = transforms.Compose([T.ToFloatTensorInZeroOne(),
                                T.Resize((200, 200)),
                                T.RandomCrop((172, 172))])
transform_test = transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((200, 200)),
    T.CenterCrop((172, 172))])

hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames, frame_rate=5,
                                           step_between_clips=clip_steps, fold=1, train=True,
                                           transform=transform, num_workers=2)

hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames, frame_rate=5,
                                          step_between_clips=clip_steps, fold=1, train=False,
                                          transform=transform_test, num_workers=2)

train_loader = DataLoader(hmdb51_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(hmdb51_test, batch_size=batch_size, shuffle=False)
