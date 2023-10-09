from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
import sys

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        #self.corr_weights = torch.nn.Parameter(torch.empty(
        #        1024, 1024, requires_grad=True).type(torch.cuda.FloatTensor))

        self.encoder1 = nn.Linear(88, 128)
        self.encoder2 = nn.Linear(168, 128)

        self.affine_a = nn.Linear(256, 128, bias=False)  # 8 before 128
        self.affine_v = nn.Linear(256, 128, bias=False)  # 8 before 128

        self.W_a = nn.Linear(128, 32, bias=False)
        self.W_v = nn.Linear(128, 32, bias=False)
        self.W_ca = nn.Linear(128, 32, bias=False)
        self.W_cv = nn.Linear(128, 32, bias=False)

        self.W_ha = nn.Linear(32, 8, bias=False)
        self.W_hv = nn.Linear(32, 8, bias=False)

        self.encoder3 = nn.Linear(128, 8)  # I created that encoder to match
        self.encoder4 = nn.Linear(128, 8)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.regressor = nn.Sequential(nn.Linear(16, 128),  # 640 before
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

    #def first_init(self):
    #    nn.init.xavier_normal_(self.corr_weights)

    def forward(self, f1_norm, f2_norm):
        #f1 = f1.squeeze(1)  # o Dataloader já remove a dimensão de tamanho 1, automaticamente 
        #f2 = f2.squeeze(1)  # o Dataloader já remove a dimensão de tamanho 1
        #print(f1_norm.shape)  #[32,88]
        #print(f2_norm.shape)  #[32,168]

        f1_norm = F.normalize(f1_norm, p=2, dim=1, eps=1e-12)  # dim=1->features, dim=0->batch  
        f2_norm = F.normalize(f2_norm, p=2, dim=1, eps=1e-12)  # ======== VALORES NAN ESTÃO SENDO GERADOS AQUI E PROPAGADOS PARA visfts =========

        fin_audio_features = []
        fin_visual_features = []
        sequence_outs = []

        for i in range(f1_norm.shape[0]):  # shape[0]=32
            audfts = f1_norm[i,:]#.transpose(0,1)
            visfts = f2_norm[i,:]#.transpose(0,1)  # ======== VALORES NAN ESTÃO SENDO GERADOS AQUI =========

            aud_fts = self.encoder1(audfts)  # aud_fts.shape=128
            vis_fts = self.encoder2(visfts)  # vis_fts.shape=128

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 0)  # joint representation of A-V features -> J  shape=256
            a_t = self.affine_a(aud_vis_fts) # input_dimension=256 (256->128)
            att_aud = aud_fts * a_t  # att_aud = X_a^T W_ja J = aud_fts^T W_ja J dim=128
            audio_att = self.tanh(torch.div(att_aud, math.sqrt(aud_vis_fts.shape[0])))  # joint correlation matrix (X_a, J) -> C_a (equation 1)
            # audio_att.shape(128)

            aud_vis_fts = torch.cat((aud_fts, vis_fts), 0)
            v_t = self.affine_v(aud_vis_fts)
            att_vis = vis_fts * v_t
            vis_att = self.tanh(torch.div(att_vis, math.sqrt(aud_vis_fts.shape[0])))  # C_v (equation 2)
            #vis_att.shape(128)

            H_a = self.relu(self.W_ca(audio_att) + self.W_a(aud_fts))  # attention matrix/map -> H_a (equation 3) -> shape=32
            H_v = self.relu(self.W_cv(vis_att) + self.W_v(vis_fts))  # attention matrix/map -> H_v (equation 4) -> shape=32

            att_audio_features = self.W_ha(H_a) + self.encoder3(aud_fts)  # X_att,a (equation 5) -> shape=8
            att_visual_features = self.W_hv(H_v) + self.encoder4(vis_fts)  # X_att,v (equation 6) -> shape=8

            #a1 = torch.matmul(aud_fts.transpose(0,1), self.corr_weights)
            #cc_mat = torch.matmul(a1, vis_fts)

            #audio_att = F.softmax(cc_mat, dim=1)
            #visual_att = F.softmax(cc_mat.transpose(0,1), dim=1)

            #atten_audiofeatures = torch.matmul(aud_fts, audio_att)
            #atten_visualfeatures = torch.matmul(vis_fts, visual_att)

            #added_atten_audiofeatures = aud_fts.add(atten_audiofeatures)
            #added_atten_visualfeatures = vis_fts.add(atten_visualfeatures)

            #### apply tanh on features

            ### apply same dimensions
            #att_audio_features = self.tanh(atten_audiofeatures)
            #att_visual_features = self.tanh(atten_visualfeatures)

            audiovisualfeatures = torch.cat((att_audio_features, att_visual_features), 0)  # X_att (equation_7) -> shape=16
            outs = self.regressor(audiovisualfeatures) #.transpose(0,1))

            #seq_outs, _ = torch.max(outs,0)
            #print(seq_outs)
            sequence_outs.append(outs)
            fin_audio_features.append(att_audio_features)
            fin_visual_features.append(att_visual_features)
        final_aud_feat = torch.stack(fin_audio_features)
        final_vis_feat = torch.stack(fin_visual_features)
        final_outs = torch.stack(sequence_outs)
        return final_outs #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)