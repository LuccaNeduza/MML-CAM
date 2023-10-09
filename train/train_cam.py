from __future__ import print_function
import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
from train.utils import set_lr
from train.metrics_cam import ccc
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

learning_rate_decay_start = 5  # 50
learning_rate_decay_every = 2 # 5
learning_rate_decay_rate = 0.8 # 0.9
total_epoch = 30
lr = 0.001
scaler = torch.cuda.amp.GradScaler()

def train(train_loader, criterion, optimizer, epoch, cam):
	device = ("cuda" if torch.cuda.is_available() else "mps" 
                  if torch.backends.mps.is_available() else "cpu")
	print(f"Using {device} device")

	print('\nEpoch: %d' % epoch)
	global Train_acc
	cam.train()

	train_loss = 0
	correct = 0
	total = 0
	running_loss = 0
	running_accuracy = 0
	out = []
	tar = []

	if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
		frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
		decay_factor = learning_rate_decay_rate ** frac
		current_lr = lr * decay_factor
		set_lr(optimizer, current_lr)  # set the decayed rate
	else:
		current_lr = lr
	print('learning_rate: %s' % str(current_lr))
	logging.info("Learning rate")
	logging.info(current_lr)

	for batch_idx, (arousal_audio_feature, arousal_video_feature, arousal_label) in tqdm(enumerate(train_loader), 
																					  total=len(train_loader), position=0, leave=True):

		optimizer.zero_grad(set_to_none=True)
		audiodata = arousal_audio_feature.to(device)
		videodata = arousal_video_feature.to(device)
		labels = arousal_label.to(device)

		with torch.cuda.amp.autocast():
			audiovisual_outs = cam(audiodata, videodata)  # shape->[32,1]
			
			outputs = audiovisual_outs.view(-1, audiovisual_outs.shape[0]*audiovisual_outs.shape[1])  # shape=[1,32]
			targets = labels#.view(-1, labels.shape[0]*labels.shape[1])#.cuda()  # labels.shape -> 32

			loss = criterion(outputs, targets)
			print(f'loss_calculated={loss}')
		
		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		out = np.concatenate([out, outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, targets.squeeze(0).detach().cpu().numpy()])

		if torch.isnan(loss):
			print(f'outputs={outputs}')
			print(f'targets={targets}')
			print(f'loss={loss}')
			sys.exit()

	if (len(tar) > 1):
		train_acc = ccc(out, tar)
	else:
		train_acc = 0
	print("Train Accuracy")
	print(train_acc)

	return (loss), (train_acc)