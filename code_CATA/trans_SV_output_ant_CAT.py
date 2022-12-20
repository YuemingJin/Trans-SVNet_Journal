#some code adapted from https://github.com/YuemingJin/MTRCNet-CL
# and https://github.com/YuemingJin/TMRNet

import torch
from torch import optim
from torch import nn
import numpy as np
import pickle, time
import random
from sklearn import metrics
import copy
import mstcn
from transformer2_3_1 import Transformer2_3_1
import os, subprocess
import os
from pandas import read_csv

torch.set_printoptions(threshold=np.inf)

#os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#    "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    # train_paths_19 = train_test_paths_labels[0]
    train_paths_80 = train_test_paths_labels[0]
    val_paths_80 = train_test_paths_labels[1]
    # train_labels_19 = train_test_paths_labels[3]
    train_labels_80 = train_test_paths_labels[2]
    val_labels_80 = train_test_paths_labels[3]
    # train_num_each_19 = train_test_paths_labels[6]
    train_num_each_80 = train_test_paths_labels[4]
    val_num_each_80 = train_test_paths_labels[5]

    test_paths_80 = train_test_paths_labels[6]
    test_labels_80 = train_test_paths_labels[7]
    test_num_each_80 = train_test_paths_labels[8]

    # print('train_paths_19  : {:6d}'.format(len(train_paths_19)))
    # print('train_labels_19 : {:6d}'.format(len(train_labels_19)))
    print('train_paths_80  : {:6d}'.format(len(train_paths_80)))
    print('train_labels_80 : {:6d}'.format(len(train_labels_80)))
    print('valid_paths_80  : {:6d}'.format(len(val_paths_80)))
    print('valid_labels_80 : {:6d}'.format(len(val_labels_80)))

    # train_labels_19 = np.asarray(train_labels_19, dtype=np.int64)
    train_labels_80 = np.asarray(train_labels_80, dtype=np.float64)
    val_labels_80 = np.asarray(val_labels_80, dtype=np.float64)
    test_labels_80 = np.asarray(test_labels_80, dtype=np.float64)

    train_start_vidx = []
    count = 0
    for i in range(len(train_num_each_80)):
        train_start_vidx.append(count)
        count += train_num_each_80[i]

    val_start_vidx = []
    count = 0
    for i in range(len(val_num_each_80)):
        val_start_vidx.append(count)
        count += val_num_each_80[i]

    test_start_vidx = []
    count = 0
    for i in range(len(test_num_each_80)):
        test_start_vidx.append(count)
        count += test_num_each_80[i]

    return train_labels_80, train_num_each_80, train_start_vidx, val_labels_80, val_num_each_80, val_start_vidx,\
           test_labels_80, test_num_each_80, test_start_vidx, train_paths_80, val_paths_80, test_paths_80

def get_long_feature(start_index, lfb, LFB_length):
    long_feature = []
    long_feature_each = []
    # 上一个存在feature的index
    for k in range(LFB_length):
        LFB_index = (start_index + k)
        LFB_index = int(LFB_index)
        long_feature_each.append(lfb[LFB_index])
    long_feature.append(long_feature_each)
    return long_feature

train_labels_80, train_num_each_80, train_start_vidx,\
    val_labels_80, val_num_each_80, val_start_vidx,\
    test_labels_80, test_num_each_80, test_start_vidx, train_path80, val_path80, test_path80= get_data('./train_val_paths_labels_CAT_ant.pkl')      

class Transformer(nn.Module):
    def __init__(self, mstcn_f_maps, mstcn_f_dim, out_features, len_q):
        super(Transformer, self).__init__()
        self.num_f_maps = mstcn_f_maps  # 32
        self.dim = mstcn_f_dim  # 2048
        self.num_classes = out_features  # 7
        self.len_q = len_q

        self.transformer = Transformer2_3_1(d_model=out_features, d_ff=mstcn_f_maps, d_k=mstcn_f_maps,
                                        d_v=mstcn_f_maps, n_layers=1, n_heads=8, len_q = sequence_length)
        self.fc = nn.Linear(mstcn_f_dim, out_features, bias=False)


    def forward(self, x, long_feature):
        out_features = x.transpose(1,2)
        inputs = []
        for i in range(out_features.size(1)):
            if i<self.len_q-1:
                input = torch.zeros((1, self.len_q-1-i, self.num_classes)).cuda()
                input = torch.cat([input, out_features[:, 0:i+1]], dim=1)
            else:
                input = out_features[:, i-self.len_q+1:i+1]
            inputs.append(input)
        inputs = torch.stack(inputs, dim=0).squeeze(1)
        feas = torch.tanh(self.fc(long_feature).transpose(0,1))
        output = self.transformer(inputs, feas)
        #output = output.transpose(1,2)
        #output = self.fc(output)
        return output

with open("./LFB/g_LFB50_train_ant_CAT.pkl", 'rb') as f:
    g_LFB_train = pickle.load(f)

with open("./LFB/g_LFB50_val_ant_CAT.pkl", 'rb') as f:
    g_LFB_val = pickle.load(f)

with open("./LFB/g_LFB50_test_ant_CAT.pkl", 'rb') as f:
    g_LFB_test = pickle.load(f)

print("load completed")

print("g_LFB_train shape:", g_LFB_train.shape)
print("g_LFB_val shape:", g_LFB_val.shape)
print("g_LFB_test shape:", g_LFB_test.shape)

out_features = 19*2
num_workers = 3
batch_size = 1
mstcn_causal_conv = True
learning_rate = 1e-4
min_epochs = 12
max_epochs = 1
mstcn_layers = 8
mstcn_f_maps = 32
mstcn_f_dim= 2048
mstcn_stages = 2
horizon = 1
downsample_rate = 6

path_origin = './output_txt/'
val_dir = os.path.join(path_origin, 'cat_val/')
test_dir = os.path.join(path_origin, 'cat_test/')
ground_true_path = '../CATARACTS/ground_truth/CATARACTS_2020/Phase/all_gt/'

sequence_length = 30

seed = 1
print("Random Seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")


criterion_phase = nn.CrossEntropyLoss()
criterion_phase1 = nn.CrossEntropyLoss()

criterion_reg = nn.SmoothL1Loss(reduction='mean')
criterion_cls = nn.CrossEntropyLoss()

model = mstcn.MultiStageModel_S(mstcn_stages, mstcn_layers, mstcn_f_maps, mstcn_f_dim, out_features, mstcn_causal_conv)
model_path = './best_model/TeCNO_ant_CAT/'
model_name = 'TeCNO50_epoch_11'
model.load_state_dict(torch.load(model_path+model_name+'.pth'))
if use_gpu:
    model.cuda()
model.eval()

model1 = Transformer(mstcn_f_maps, mstcn_f_dim, out_features, sequence_length)
model1_path = './best_model/TeCNO_ant_t_CAT/'
model1_name = 'TeCNO50_trans1_3_5_1_length_30_epoch_1_train_8443_val_6968'
model1.load_state_dict(torch.load(model1_path+model1_name+'.pth'))
if use_gpu:
    model1.cuda()
model1.eval()


best_model_wts = copy.deepcopy(model1.state_dict())
best_val_accuracy_phase = 0.0
correspond_train_acc_phase = 0.0
best_epoch = 0

train_we_use_start_idx_80 = [x for x in range(25)]
val_we_use_start_idx_80 = [x for x in range(5)]
test_we_use_start_idx_80 = [x for x in range(20)]

for epoch in range(max_epochs):
    torch.cuda.empty_cache()
    random.shuffle(train_we_use_start_idx_80)
    train_idx_80 = []
    model1.train()
    train_loss_phase = 0.0
    train_corrects_phase = 0
    batch_progress = 0.0
    running_loss_phase = 0.0
    minibatch_correct_phase = 0.0
    train_start_time = time.time()

    # Sets the module in evaluation mode.
    model.eval()
    model1.eval()
    val_loss_phase = 0.0
    val_corrects_phase = 0
    val_start_time = time.time()
    val_progress = 0
    val_all_preds_phase = []
    val_all_labels_phase = []
    val_acc_each_video = []
    in_MAE = []
    pMAE = []
    eMAE = []
    predict_phase_ant_all = []
    gt_phase_ant_all = []

    with torch.no_grad():
        for i in val_we_use_start_idx_80:
            labels_phase = []
            labels_phase_ant = []
            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                labels_phase.append(val_labels_80[j][0])
                labels_phase_ant.append(val_labels_80[j][1: 20])

            labels_phase = torch.LongTensor(np.array(labels_phase))
            labels_phase_ant = torch.Tensor(np.array(labels_phase_ant))
            if use_gpu:
                labels_phase = labels_phase.to(device)
                labels_phase_ant = labels_phase_ant.to(device)
            else:
                labels_phase = labels_phase
                labels_phase_ant = labels_phase_ant

            long_feature = get_long_feature(start_index=val_start_vidx[i],
                                            lfb=g_LFB_val, LFB_length=val_num_each_80[i])

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)
            video_fe = long_feature.transpose(2, 1)

            out_features = model.forward(video_fe)[-1]     # outfeature --》 from multi
            out_features = out_features.squeeze(1)

            p_all = model1(out_features.detach(), long_feature)

            p_classes1 = p_all[:,:,:19]
            p_anticipation = p_all[:,:,19:]

            p_classes = p_classes1.squeeze()
            clc_loss = criterion_phase1(p_classes, labels_phase)

            p_anticipation = p_anticipation.squeeze()

            _, preds_phase = torch.max(p_classes.data, 1)
            path_whole = val_path80[val_start_vidx[i]]
            path_s = path_whole.split('/')
            print(path_s[-2])

            gt_file = read_csv(ground_true_path+path_s[-2]+'.csv')

            #TODO ！！！！！！！！！==========================================

            f = open(val_dir+path_s[-2]+'.csv','w')
            # first_line = 1
            cnt = 0
            f.write('Frame,Steps')

            for j in range(val_start_vidx[i], val_start_vidx[i] + val_num_each_80[i]):
                # if first_line:
                #     f.write('Frame'+'\t'+'Phase'+'\t'+'Step'+'\t'+'Verb_Left'+'\t'+'Verb_Right'+'\n')
                #     first_line = 0
                             
                p_cpu = preds_phase.cpu()
                p_num = p_cpu.numpy()

                # ~~~~~~ upsample ~~~~
                if cnt>0:
                    for index in range((cnt-1)*downsample_rate+2,cnt*downsample_rate+1):
                        f.write('\n')
                        f.write(str(index)+',')
                        f.write(str(p_num[cnt]))

                # print(p_num[cnt])
                f.write('\n')
                f.write(str(cnt*downsample_rate+1)+',')
                f.write(str(p_num[cnt]))
                cnt = cnt + 1

            for index in range((cnt-1)*downsample_rate+2,len(gt_file)+1):
                f.write('\n')
                f.write(str(index)+',')
                f.write(str(p_num[-1]))
                
             # ~~~~~~ upsample ~~~~

            f.close()
            
            loss_phase = criterion_phase1(p_classes, labels_phase)

            val_loss_phase += loss_phase.data.item()

            val_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            val_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data))/val_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                val_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                val_all_labels_phase.append(int(labels_phase.data.cpu()[j]))

            outputs_phase_ant = p_anticipation
            for j in range(len(outputs_phase_ant)):
                predict_phase_ant_all.append(outputs_phase_ant.data.cpu().numpy()[j])
            for j in range(len(labels_phase_ant)):
                gt_phase_ant_all.append(labels_phase_ant.data.cpu().numpy()[j])


            val_progress += 1
            if val_progress * batch_size >= len(val_we_use_start_idx_80):
                percent = 100.0
                print('Val progress: %s [%d/%d]' % (str(percent) + '%', len(val_we_use_start_idx_80),
                                                    len(val_we_use_start_idx_80)), end='\n')
            else:
                percent = round(val_progress * batch_size / len(val_we_use_start_idx_80) * 100, 2)
                print('Val progress: %s [%d/%d]' % (str(percent) + '%', val_progress * batch_size, len(val_we_use_start_idx_80)),
                      end='\n')

    predict_phase_ant_all = np.array(predict_phase_ant_all).transpose(1,0)
    gt_phase_ant_all = np.array(gt_phase_ant_all).transpose(1,0)

    for y, t in zip(predict_phase_ant_all, gt_phase_ant_all):

        inside_horizon = (t > 0.0) & (t < 1.0)
        anticipating = (y > 1*.1) & (y < 1*.9)
        e_anticipating = (t < 1*.1) & (t > 0.0)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon]*horizon-t[inside_horizon]*horizon))            
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating]*horizon-t[anticipating]*horizon))                
        if not np.isnan(pMAE_ins):            
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating]*horizon-t[e_anticipating]*horizon))                
        if not np.isnan(eMAE_ins):              
            eMAE.append(eMAE_ins)

    in_MAE_val = np.mean(in_MAE)
    pMAE_val = np.mean(pMAE) 
    eMAE_val = np.mean(eMAE)


    val_elapsed_time = time.time() - val_start_time
    val_accuracy_phase = float(val_corrects_phase) / len(val_labels_80)
    val_acc_video = np.mean(val_acc_each_video)
    val_average_loss_phase = val_loss_phase

    val_recall_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_jaccard_phase = metrics.jaccard_score(val_all_labels_phase, val_all_preds_phase, average='macro')
    val_precision_each_phase = metrics.precision_score(val_all_labels_phase, val_all_preds_phase, average=None)
    val_recall_each_phase = metrics.recall_score(val_all_labels_phase, val_all_preds_phase, average=None)

    test_progress = 0
    test_corrects_phase = 0
    test_all_preds_phase = []
    test_all_labels_phase = []
    test_acc_each_video = []
    in_MAE = []
    pMAE = []
    eMAE = []
    predict_phase_ant_all = []
    gt_phase_ant_all = []
    test_start_time = time.time()

    with torch.no_grad():
        for i in test_we_use_start_idx_80:
            labels_phase = []
            labels_phase_ant = []
            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                labels_phase.append(test_labels_80[j][0])
                labels_phase_ant.append(test_labels_80[j][1: 20])
            labels_phase = torch.LongTensor(np.array(labels_phase))
            labels_phase_ant = torch.Tensor(np.array(labels_phase_ant))
            if use_gpu:
                labels_phase = labels_phase.to(device)
                labels_phase_ant = labels_phase_ant.to(device)
            else:
                labels_phase = labels_phase
                labels_phase_ant = labels_phase_ant

            long_feature = get_long_feature(start_index=test_start_vidx[i],
                                            lfb=g_LFB_test, LFB_length=test_num_each_80[i])

            long_feature = (torch.Tensor(np.array(long_feature))).to(device)
            video_fe = long_feature.transpose(2, 1)

            out_features = model.forward(video_fe)[-1]
            out_features = out_features.squeeze(1)
            p_all = model1(out_features.detach(), long_feature)

            p_classes1 = p_all[:,:,:19]
            p_anticipation = p_all[:,:,19:]

            p_classes = p_classes1.squeeze()    # final decision
            clc_loss = criterion_phase1(p_classes, labels_phase)  #交叉熵

            p_anticipation = p_anticipation.squeeze()

            _, preds_phase = torch.max(p_classes.data, 1)
            path_whole = test_path80[test_start_vidx[i]]

            path_s = path_whole.split('/')
            print(path_s[-2])

            gt_file = read_csv(ground_true_path+path_s[-2]+'.csv')
            # print(len(gt_file))

            #TODO ！！！！！！！！！==========================================

            f = open(test_dir+path_s[-2]+'.csv','w')
            # first_line = 1
            cnt = 0
            f.write('Frame,Steps')

            for j in range(test_start_vidx[i], test_start_vidx[i] + test_num_each_80[i]):
                # if first_line:
                #     f.write('Frame' + '\t' + 'Phase' + '\t' + 'Step' + '\t' + 'Verb_Left' + '\t' + 'Verb_Right' + '\n')
                #     first_line = 0
                p_cpu = preds_phase.cpu()
                p_num = p_cpu.numpy()

                # ~~~~~~ upsample ~~~~
                if cnt>0:
                    for index in range((cnt-1)*downsample_rate+2,cnt*downsample_rate+1):
                        f.write('\n')
                        f.write(str(index)+',')
                        f.write(str(p_num[cnt]))

                # print(p_num[cnt])
                f.write('\n')
                f.write(str(cnt*downsample_rate+1)+',')
                f.write(str(p_num[cnt]))
                cnt = cnt + 1

            for index in range((cnt-1)*downsample_rate+2,len(gt_file)+1):
                f.write('\n')
                f.write(str(index)+',')
                f.write(str(p_num[-1]))

            # ~~~~~~ upsample ~~~~
            f.close()
            in_MAE_temp = []
            pMAE_temp = []
            eMAE_temp = []

            for y, t in zip(p_anticipation.data.cpu().numpy().transpose(1,0), labels_phase_ant.data.cpu().numpy().transpose(1,0)):

                inside_horizon = (t > 0.0) & (t < 1.0)
                anticipating = (y > 1*.1) & (y < 1*.9)
                e_anticipating = (t < 1*.1) & (t > 0.0)

                in_MAE_ins = np.mean(np.abs(y[inside_horizon]*horizon-t[inside_horizon]*horizon))            
                if not np.isnan(in_MAE_ins):
                    in_MAE_temp.append(in_MAE_ins)

                pMAE_ins = np.mean(np.abs(y[anticipating]*horizon-t[anticipating]*horizon))                
                if not np.isnan(pMAE_ins):            
                    pMAE_temp.append(pMAE_ins)

                eMAE_ins = np.mean(np.abs(y[e_anticipating]*horizon-t[e_anticipating]*horizon))                
                if not np.isnan(eMAE_ins):              
                    eMAE_temp.append(eMAE_ins) 

            print('--',np.mean(in_MAE_temp),np.mean(pMAE_temp),np.mean(eMAE_temp))         

            test_corrects_phase += torch.sum(preds_phase == labels_phase.data)
            test_acc_each_video.append(float(torch.sum(preds_phase == labels_phase.data)) / test_num_each_80[i])
            # TODO

            for j in range(len(preds_phase)):
                test_all_preds_phase.append(int(preds_phase.data.cpu()[j]))
            for j in range(len(labels_phase)):
                test_all_labels_phase.append(int(labels_phase.data.cpu()[j]))
            
            outputs_phase_ant = p_anticipation
            for j in range(len(outputs_phase_ant)):
                predict_phase_ant_all.append(outputs_phase_ant.data.cpu().numpy()[j])
            for j in range(len(labels_phase_ant)):
                gt_phase_ant_all.append(labels_phase_ant.data.cpu().numpy()[j])

            test_progress += 1
            if test_progress * batch_size >= len(test_we_use_start_idx_80):
                percent = 100.0
                print('Test progress: %s [%d/%d]' % (str(percent) + '%', len(test_we_use_start_idx_80),
                                                    len(test_we_use_start_idx_80)), end='\n')
            else:
                percent = round(test_progress * batch_size / len(test_we_use_start_idx_80) * 100, 2)
                print('Test progress: %s [%d/%d]' % (
                str(percent) + '%', test_progress * batch_size, len(test_we_use_start_idx_80)),
                      end='\n')


    predict_phase_ant_all = np.array(predict_phase_ant_all).transpose(1,0)
    gt_phase_ant_all = np.array(gt_phase_ant_all).transpose(1,0)

    for y, t in zip(predict_phase_ant_all, gt_phase_ant_all):

        inside_horizon = (t > 0.0) & (t < 1.0)
        anticipating = (y > 1*.1) & (y < 1*.9)
        e_anticipating = (t < 1*.1) & (t > 0.0)

        in_MAE_ins = np.mean(np.abs(y[inside_horizon]*horizon-t[inside_horizon]*horizon))            
        if not np.isnan(in_MAE_ins):
            in_MAE.append(in_MAE_ins)

        pMAE_ins = np.mean(np.abs(y[anticipating]*horizon-t[anticipating]*horizon))                
        if not np.isnan(pMAE_ins):            
            pMAE.append(pMAE_ins)

        eMAE_ins = np.mean(np.abs(y[e_anticipating]*horizon-t[e_anticipating]*horizon))                
        if not np.isnan(eMAE_ins):              
            eMAE.append(eMAE_ins)

    in_MAE_test = np.mean(in_MAE)
    pMAE_test = np.mean(pMAE)
    eMAE_test = np.mean(eMAE)


    test_accuracy_phase = float(test_corrects_phase) / len(test_labels_80)
    test_acc_video = np.mean(test_acc_each_video)
    test_elapsed_time = time.time() - test_start_time

    print('epoch: {:4d}'
          ' valid in: {:2.0f}m{:2.0f}s'
          ' valid loss(phase): {:4.4f}'
          ' valid accu(phase): {:.4f}'
          ' valid accu(video): {:.4f}'
          ' valid in_MAE(phase): {:.4f}'
          ' valid pMAE(phase): {:.4f}'
          ' valid eMAE(phase): {:.4f}'
          ' test in: {:2.0f}m{:2.0f}s'
          ' test accu(phase): {:.4f}'
          ' test accu(video): {:.4f}'
          ' test in_MAE(phase): {:.4f}'
          ' test pMAE(phase): {:.4f}'
          ' test eMAE(phase): {:.4f}'
          .format(epoch,
                  val_elapsed_time // 60,
                  val_elapsed_time % 60,
                  val_average_loss_phase,
                  val_accuracy_phase,
                  val_acc_video,
                  in_MAE_val,
                  pMAE_val,
                  eMAE_val,
                  test_elapsed_time // 60,
                  test_elapsed_time % 60,
                  test_accuracy_phase,
                  test_acc_video,
                  in_MAE_test,
                  pMAE_test,
                  eMAE_test))

    print("val_precision_each_phase:", val_precision_each_phase)
    print("val_recall_each_phase:", val_recall_each_phase)
    print("val_precision_phase", val_precision_phase)
    print("val_recall_phase", val_recall_phase)
    print("val_jaccard_phase", val_jaccard_phase)



