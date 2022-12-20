
import os
import numpy as np
import pickle
import csv

root_dir = '/research/dept6/yhlong/Phase/CATARACTS/'
downsample_rate = 6

# ['test01', 'test02', 'test03', 'test04', 'test05', 'test06', 'test07', 'test08', 'test09', 'test10', 
# 'test11', 'test12', 'test13', 'test14', 'test15', 'test16', 'test17', 'test18', 'test19', 'test20', 
# 'test21', 'test22', 'test23', 'test24', 'test25', 
# 'train01', 'train02', 'train03', 'train04', 'train05', 'train06', 'train07', 'train08', 'train09', 'train10', 
# 'train11', 'train12', 'train13', 'train14', 'train15', 'train16', 'train17', 'train18', 'train19', 'train20', 
# 'train21', 'train22', 'train23', 'train24', 'train25']

vids_for_training = [i for i in range(25,50)]
vids_for_val = [0,6,13,15,18]
vids_for_test = [i if i not in vids_for_val else None for i in range(0,25)]
vids_for_test = list(filter(None, vids_for_test))

print(vids_for_training)
print(vids_for_val)
print(vids_for_test)

img_dir2 = os.path.join(root_dir, 'images')
phase_dir2 = os.path.join(root_dir, 'ground_truth', 'CATARACTS_2020', 'Phase', 'all_gt')
phase_ant_dir2 = os.path.join(root_dir, 'ground_truth', 'CATARACTS_2020', 'Phase_ant')

print(root_dir)
print(img_dir2)
print(phase_dir2)
print(phase_ant_dir2)

# CATARACTS==================
def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort(key=lambda x: os.path.basename(x))
    return file_names, file_paths

def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: os.path.splitext(x)[0])
    file_paths.sort(key=lambda x: os.path.splitext(os.path.basename(x))[0])
    return file_names, file_paths

# CATARACTS==================

img_dir_names2, img_dir_paths2 = get_dirs(img_dir2)
phase_file_names2, phase_file_paths2 = get_files(phase_dir2)
phase_ant_file_names2, phase_ant_file_paths2 = get_files(phase_ant_dir2)

print(img_dir_names2)
print(img_dir_paths2)

print(phase_file_names2)
print(phase_file_paths2)

print(phase_ant_file_names2)
print(phase_ant_file_paths2)

all_info_all_test = []

for j in vids_for_test:

    phase_file = csv.reader(open(phase_file_paths2[j]), delimiter=',')
    phase_ant_file = open(phase_ant_file_paths2[j])

    video_num_file = os.path.splitext(os.path.basename(phase_file_paths2[j]))[0]
    video_num_dir = os.path.basename(img_dir_paths2[j])

    print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", 1)

    info_all = []
    next(phase_file, None)
    for phase_line in phase_file:
        phase_split = phase_line
        if (int(phase_split[0])-1) % 1 == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], str(int(phase_split[0])-1) + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(int(phase_split[1]))
            info_all.append(info_each)

    count_phase_ant = 0
    count = 0
    for phase_ant_line in phase_ant_file:
        phase_ant_split = phase_ant_line.split()
        if count % 1 == 0:
            for index_int in range(19):
                info_all[count_phase_ant].append(float(phase_ant_split[index_int]))
            count_phase_ant += 1
        count += 1

    all_info_all_test.append(info_all)

with open('./CATARACTS_test.pkl', 'wb') as f:
    pickle.dump(all_info_all_test, f)

all_info_all2 = []

for j in range(len(phase_file_names2)):

    phase_file = csv.reader(open(phase_file_paths2[j]), delimiter=',')
    phase_ant_file = open(phase_ant_file_paths2[j])

    video_num_file = os.path.splitext(os.path.basename(phase_file_paths2[j]))[0]
    video_num_dir = os.path.basename(img_dir_paths2[j])

    print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    next(phase_file, None)
    for phase_line in phase_file:
        phase_split = phase_line
        if (int(phase_split[0])-1) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], str(int(phase_split[0])-1) + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(int(phase_split[1]))
            info_all.append(info_each)

    count_phase_ant = 0
    count = 0
    for phase_ant_line in phase_ant_file:
        phase_ant_split = phase_ant_line.split()
        if count % downsample_rate == 0:
            for index_int in range(19):
                info_all[count_phase_ant].append(float(phase_ant_split[index_int]))
            count_phase_ant += 1
        count += 1

    all_info_all2.append(info_all)

with open('./CATARACTS.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)

with open('./CATARACTS.pkl', 'rb') as f:
    all_info_80 = pickle.load(f)




train_file_paths_80 = []
test_file_paths_80 = []
val_file_paths_80 = []
val_labels_80 = []
train_labels_80 = []
test_labels_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

stat = np.zeros(19).astype(int)
for i in vids_for_training:
    train_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        train_file_paths_80.append(all_info_80[i][j][0])
        train_labels_80.append(all_info_80[i][j][1:])
        stat[all_info_80[i][j][1]] += 1

print(len(train_file_paths_80))
print(len(train_labels_80))
print(stat)
print(np.max(np.array(train_labels_80)[:, 0]))
print(np.min(np.array(train_labels_80)[:, 0]))

for i in vids_for_val:
    val_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        val_file_paths_80.append(all_info_80[i][j][0])
        val_labels_80.append(all_info_80[i][j][1:])

for i in vids_for_test:
    test_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        test_file_paths_80.append(all_info_80[i][j][0])
        test_labels_80.append(all_info_80[i][j][1:])

print(len(val_file_paths_80))
print(len(val_labels_80))


train_val_test_paths_labels = []
train_val_test_paths_labels.append(train_file_paths_80)
train_val_test_paths_labels.append(val_file_paths_80)

train_val_test_paths_labels.append(train_labels_80)
train_val_test_paths_labels.append(val_labels_80)

train_val_test_paths_labels.append(train_num_each_80)
train_val_test_paths_labels.append(val_num_each_80)


train_val_test_paths_labels.append(test_file_paths_80)
train_val_test_paths_labels.append(test_labels_80)
train_val_test_paths_labels.append(test_num_each_80)


with open('train_val_paths_labels_CAT_ant.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

print('Done')
print()