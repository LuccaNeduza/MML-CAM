from __future__ import print_function
import argparse
import os
import shutil
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
#import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#from torchsummary import summary
import torchvision.models as models
# from models import *
from collections import OrderedDict
from torch.autograd import Variable
# import scipy as sp
from scipy import signal
import logging
# import models.resnet as ResNet
import utils
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import sys
from metrics_cam import ccc

import math
#import wandb


def validate(val_loader, visual_model, audio_model, criterion, epoch, cam):
	# switch to evaluate mode
	global Val_acc
	global best_Val_acc
	global best_Val_acc_epoch
	#model.eval()
	audio_model.eval()
	visual_model.eval()
	cam.eval()

	PrivateTest_loss = 0
	correct = 0
	total = 0
	running_val_loss = 0
	running_val_accuracy = 0

	out = []
	tar = []
	#torch.cuda.synchronize()
	#t7 = time.time()

	for batch_idx, (arousal_audio_feature, arousal_video_feature, arousal_label) in tqdm(enumerate(val_loader)):


		audiodata = arousal_audio_feature.cuda()
		videodata = arousal_video_feature.cuda()
		labels = arousal_label.cuda()

		#torch.cuda.synchronize()
		#t9 = time.time()

		with torch.no_grad():
			audiovisual_outs = cam(audiodata, videodata)
			outputs = audiovisual_outs.view(-1, audiovisual_outs.shape[0]*audiovisual_outs.shape[1])
			targets = labels.view(-1, labels.shape[0]*labels.shape[1]).cuda()

		val_loss = criterion(outputs, targets)

		out = np.concatenate([out, outputs.squeeze(0).detach().cpu().numpy()])
		tar = np.concatenate([tar, targets.squeeze(0).detach().cpu().numpy()])

	if (len(tar) > 1):
		Val_acc = ccc(out, tar)
	else:
		Val_acc = 0

	print("Val Accuracy")
	#wandb.log({"Val_acc": Val_acc})

	print(Val_acc)
	return val_loss, (Val_acc)
