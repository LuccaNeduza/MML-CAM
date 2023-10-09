import argparse
import warnings 
from preprocess import data_generator
from train import train
import config
from train import utils
from train import train_cam
import torch
from torch.utils.data import Dataset, DataLoader
from train import dataloader
from train.CCC import CCC
from train.cam import CAM
import logging
import os

warnings.filterwarnings('ignore')

"""
======== Pre-Processing/Training Code ========
Author: Luciana Menon
Code available on page: 'https://github.com/lucianamenon/ds-multimodal-emotion-recognition'

======== Cross-Attentional AV Fusion Model ========
Paper: A Joint Cross-Attention Model for Audio-Visual Fusion in Dimensional
Emotion Recognition
Authors: R Gnana Praveen, Wheidima Carneiro de Melo, Nasib Ullah, Haseeb Aslam1, Osama Zeeshan1, Théo
Denorme, Marco Pedersoli, Alessandro L. Koerich, Simon Bacon, Patrick Cardinal, and Eric Granger
Code available on page: 'https://github.com/praveena2j/Cross-Attentional-AV-Fusion'
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest='command')
    preprocess = subparser.add_parser('preprocess', help='data pre-processing')
    training = subparser.add_parser('train', help='train models')
    training_cam = subparser.add_parser('cam', help='Cross-Attention Model')

    training.add_argument('--seq_len', type=int, default=50)
    training.add_argument('--return_sequences', action='store_true', default=False)
    training.add_argument('--models', type=str, choices=['all', 'audio', 'video'], default='all')
    training.add_argument('--attention', type=str, choices=['all', 'cam'], default='cam')
    training.add_argument('--early_fusion', default=False)

    args = parser.parse_args()
    if args.command == 'preprocess':
        data_generator.data_generation()

    if args.command == 'train':
        seq_len = args.seq_len
        return_sequences = args.return_sequences
        model = args.models
        early_fusion = args.early_fusion
        train.run(seq_len, return_sequences, model, early_fusion)
    
    if args.command == 'cam':
        device = ("cuda" if torch.cuda.is_available() else "mps" 
                  if torch.backends.mps.is_available() else "cpu")
        
        print(f"Using {device} device")

        data = utils.load_data(config.RECOLA_PICKLE_PATH)
        train_dataloader = dataloader.AudioSet(data, arousal_fusion=True)
        train_loader = DataLoader(train_dataloader, batch_size=32, shuffle=True)
        criterion = CCC().to(device)
        cam = CAM().to(device)
        optimizer = torch.optim.Adam(cam.parameters())
        EPOCH = 10
        
        TrainingAccuracy = []
        ValidationAccuracy = []
        for epoch in range(EPOCH):
            logging.info("Epoch")
            logging.info(epoch)
            Training_loss, Training_acc = train_cam.train(train_loader, criterion, optimizer, epoch, cam)

            #luizValid_loss, Valid_acc = train.val.validate(train_loader, criterion, optimizer, epoch, cam)
            TrainingAccuracy.append(Training_acc)
            #luizValidationAccuracy.append(Valid_acc)

            logging.info('TrainingAccuracy:')
            logging.info(TrainingAccuracy)
            
            #luizlogging.info('ValidationAccuracy:')
            #luizlogging.info(ValidationAccuracy)

            """
            if Valid_acc > best_Val_acc:
                print('Saving..')
                print("best_Val_acc: %0.3f" % Valid_acc)
                state = {
                        'net': cam.state_dict(),
                        'best_Val_acc': Valid_acc,
                        'best_Val_acc_epoch': epoch,
		        }
                if not os.path.isdir(path):
                    os.mkdir(path)
                torch.save(state, os.path.join(path,'Val_model_valence_cnn_lstm_mil_64_new.t7'))
                best_Val_acc = Valid_acc
                best_Val_acc_epoch = epoch

print("best_PrivateTest_acc: %0.3f" % best_Val_acc)
print("best_PrivateTest_acc_epoch: %d" % best_Val_acc_epoch)
"""
