# Modified from https://gitlab.com/nct_tso_public/ins_ant/-/blob/master/train_test_scripts/dataloader.py
import os
import torch
import csv
import numpy as np
import matplotlib.pyplot as plt

# generates the ground truth signal over time for a single phase and a single operation
def generate_anticipation_gt_onephase(phase_code,horizon):
	# initialize ground truth signal
	anticipation = torch.zeros_like(phase_code).type(torch.FloatTensor)
	# default ground truth value is <horizon> minutes
	# (i.e. phase will not appear within next <horizon> minutes)
	anticipation_count = horizon
	# iterate through phase-presence signal backwards
	for i in torch.arange(len(phase_code)-1,-1,-1):
		# if phase is present, then set anticipation value to 0 minutes
		if phase_code[i]:
			anticipation_count = 0
		# else increase anticipation value with each (reverse) time step but clip at <horizon> minutes
		# video is sampled at 1fps, so 1 step = 1/60 minutes
		else:
			anticipation_count = min(horizon, anticipation_count + 1/1500)
		anticipation[i] = anticipation_count
	# normalize ground truth signal to values between 0 and 1
	anticipation = anticipation / horizon
	return anticipation

# generates the ground truth signal over time for a single operation
def generate_anticipation_gt(phases,horizon):
	return torch.stack([generate_anticipation_gt_onephase(phase_code,horizon) for phase_code in phases]).permute(1,0)

def plot_phase_anticipation(save_path, phase_gt, phase_pred=None):
    plt.clf()
    plt.figure(figsize=(30, 2*phase_gt.shape[-1]))

    for i in range(phase_gt.shape[-1]):

        ax1=plt.subplot(phase_gt.shape[-1],1,i+1)
        ax1.plot([x for x in range(len(phase_gt[:,i]))], phase_gt[:,i],color="red",linewidth=1)
        if phase_pred is not None:
            ax1.plot([x for x in range(len(phase_pred[:,i]))], phase_pred[:,i],color="blue",linewidth=1)
        
        plt.ylabel(str(i))
        plt.yticks([0,0.5,1], ['0','0.5','>1'])

    plt.xlabel("frame") #xlabel、ylabel：分别设置X、Y轴的标题文字。
    plt.savefig(save_path, dpi=120,bbox_inches='tight')



if __name__ == "__main__":

    CHOLEC80 = False
    # CATARACTS

    if CHOLEC80:

        annotation_path = '/research/dept6/yhlong/Phase/cholec80/phase_annotations/'
        save_path = '/research/dept6/yhlong/Phase/cholec80/phase_anticipation_annotations/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        horizon = 5 # max estimate time (mins)
        phase_dict = {}
        phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                        'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
        for i in range(len(phase_dict_key)):
            phase_dict[phase_dict_key[i]] = i

        annotation_paths = os.listdir(annotation_path)

        for file_name in annotation_paths:

            with open(annotation_path+file_name, "r") as f:
                phases = []
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)
                for i,row in enumerate(reader):
                    phases.append([1 if int(phase_dict[row[1]])==x else 0 for x in range(7)])

                phases = torch.LongTensor(phases)
                print(phases.shape)
                print(phases)            
                phases = phases.permute(1,0)


            target_reg = generate_anticipation_gt(phases, horizon=horizon)

            np.savetxt(save_path+file_name,target_reg)        

            print(target_reg.shape)
            print(target_reg)

            plot_phase_anticipation("./output/" + file_name.split(".")[0]+".png",target_reg*horizon)
        
    else:
        annotation_path = '/research/dept6/yhlong/Phase/CATARACTS/ground_truth/CATARACTS_2020/Phase/all_gt/'
        save_path = '/research/dept6/yhlong/Phase/CATARACTS/ground_truth/CATARACTS_2020/Phase_ant/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        horizon = 1 # max estimate time (mins)
        phase_dict = {}

        annotation_paths = os.listdir(annotation_path)

        for file_name in annotation_paths:

            with open(annotation_path+file_name, "r") as f:
                phases = []
                reader = csv.reader(f, delimiter=',')
                next(reader, None)
                for i,row in enumerate(reader):
                    phases.append([1 if int(row[1])==x else 0 for x in range(19)])

                phases = torch.LongTensor(phases)
                print(phases.shape)
                print(phases)            
                phases = phases.permute(1,0)


            target_reg = generate_anticipation_gt(phases, horizon=horizon)

            np.savetxt(save_path+file_name.split(".")[0]+".txt",target_reg)        

            print(target_reg.shape)
            print(target_reg)

            plot_phase_anticipation("./output/" + file_name.split(".")[0]+".png",target_reg*horizon)
