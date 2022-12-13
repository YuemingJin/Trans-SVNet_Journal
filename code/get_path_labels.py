import os
import numpy as np
import pickle

# root_dir = '/research/dept6/yhlong/Miccai19_challenge/Miccai19/'
# img_dir = os.path.join(root_dir, 'frame')
# phase_dir = os.path.join(root_dir, 'Annotation', 'phase')
# tool_dir = os.path.join(root_dir, 'Annotation', 'tool')

root_dir2 = '/research/dept6/yhlong/Phase/cholec80/'

img_dir2 = os.path.join(root_dir2, 'cutMargin')
phase_dir2 = os.path.join(root_dir2, 'phase_annotations')
tool_dir2 = os.path.join(root_dir2, 'tool_annotations')
phase_ant_dir2 = os.path.join(root_dir2, 'phase_anticipation_annotations')

# train_video_num = 8
# val_video_num = 4
# test_video_num = 2

# print(root_dir)
# print(img_dir)
# print(phase_dir)
# print(tool_dir)
print(root_dir2)
print(img_dir2)
print(phase_dir2)
print(phase_ant_dir2)

'''
# Miccai19==================
def get_dirs(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(x))
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths


def get_files(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(os.path.splitext(x)[0][9:11]))
    file_paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0][9:11]))
    return file_names, file_paths


# Miccai19==================

'''
# cholec80==================
def get_dirs2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort(key=lambda x: int(x))
    file_paths.sort(key=lambda x: int(os.path.basename(x)))
    return file_names, file_paths


def get_files2(root_dir):
    file_paths = []
    file_names = []
    for lists in os.listdir(root_dir):
        path = os.path.join(root_dir, lists)
        if not os.path.isdir(path):
            file_paths.append(path)
            file_names.append(os.path.basename(path))
    file_names.sort()
    file_paths.sort()
    return file_names, file_paths


# cholec80==================

'''
# Miccai19==================
img_dir_names, img_dir_paths = get_dirs(img_dir)
tool_file_names, tool_file_paths = get_files(tool_dir)
phase_file_names, phase_file_paths = get_files(phase_dir)
# Miccai19==================
'''

# cholec80==================
img_dir_names2, img_dir_paths2 = get_dirs2(img_dir2)
tool_file_names2, tool_file_paths2 = get_files2(tool_dir2)
phase_file_names2, phase_file_paths2 = get_files2(phase_dir2)
phase_ant_file_names2, phase_ant_file_paths2 = get_files2(phase_ant_dir2)

phase_dict = {}
phase_dict_key = ['Preparation', 'CalotTriangleDissection', 'ClippingCutting', 'GallbladderDissection',
                  'GallbladderPackaging', 'CleaningCoagulation', 'GallbladderRetraction']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i
print(phase_dict)
# cholec80==================

'''
# Miccai19==================
all_info_all = []

for j in range(len(phase_file_names)):

    tool_file = open(tool_file_paths[j])
    phase_file = open(phase_file_paths[j])
    info_all = []

    # 训练集测试集降采样为1FPS

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths[j]))[0][9:11])
    video_num_dir = int(os.path.basename(img_dir_paths[j]))
    if 16 <= video_num_file <= 20 or video_num_file == 23 or video_num_file == 24:
        downsample_rate = 50
    else:
        downsample_rate = 25

    print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", downsample_rate)

    for phase_line in phase_file:
        phase_split = phase_line.split(',')
        # TODO 储存img的路径
        # FPS1
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths[j], phase_split[0] + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(int(phase_split[1]))
            info_all.append(info_each)  # info_each一个视频的一个帧img_file + label

    count_tool = 0
    for tool_line in tool_file:
        tool_split = tool_line.split(',')

        if int(tool_split[0]) % downsample_rate == 0:
            for l in range(1, 8):
                info_all[count_tool].append(int(tool_split[l]))  # info_each一个视频的一个帧img_file + phase + tool
            count_tool += 1

    all_info_all.append(info_all)  # info_all一个视频的全部 all_info_all全部视频
# Miccai19==================
'''

# cholec80==================
all_info_all2 = []

for j in range(len(phase_file_names2)):
    downsample_rate = 25
    phase_file = open(phase_file_paths2[j])
    tool_file = open(tool_file_paths2[j])
    phase_ant_file = open(phase_ant_file_paths2[j])

    video_num_file = int(os.path.splitext(os.path.basename(phase_file_paths2[j]))[0][5:7])
    video_num_dir = int(os.path.basename(img_dir_paths2[j]))

    print("video_num_file:", video_num_file, "video_num_dir:", video_num_dir, "rate:", downsample_rate)

    info_all = []
    first_line = True
    for phase_line in phase_file:
        phase_split = phase_line.split()
        if first_line:
            first_line = False
            continue
        if int(phase_split[0]) % downsample_rate == 0:
            info_each = []
            img_file_each_path = os.path.join(img_dir_paths2[j], phase_split[0] + '.jpg')
            info_each.append(img_file_each_path)
            info_each.append(phase_dict[phase_split[1]])
            info_all.append(info_each)
    # TODO
    count_tool = 0
    first_line = True
    for tool_line in tool_file:
        tool_split = tool_line.split()
        if first_line:
            first_line = False
            continue
        if int(tool_split[0]) % downsample_rate == 0:
            info_all[count_tool].append(int(tool_split[0 + 1]))
            info_all[count_tool].append(int(tool_split[4 + 1]))
            info_all[count_tool].append(int(tool_split[2 + 1]))
            info_all[count_tool].append(int(tool_split[3 + 1]))
            info_all[count_tool].append(int(tool_split[5 + 1]))
            info_all[count_tool].append(int(tool_split[6 + 1]))
            info_all[count_tool].append(int(0))
            count_tool += 1
        if count_tool == len(info_all) - 1:
            info_all[count_tool].append(int(tool_split[0 + 1]))
            info_all[count_tool].append(int(tool_split[4 + 1]))
            info_all[count_tool].append(int(tool_split[2 + 1]))
            info_all[count_tool].append(int(tool_split[3 + 1]))
            info_all[count_tool].append(int(tool_split[5 + 1]))
            info_all[count_tool].append(int(tool_split[6 + 1]))
            info_all[count_tool].append(int(0))

    count_phase_ant = 0
    count = 0
    for phase_ant_line in phase_ant_file:
        phase_ant_split = phase_ant_line.split()
        if count % downsample_rate == 0:
            info_all[count_phase_ant].append(float(phase_ant_split[0]))
            info_all[count_phase_ant].append(float(phase_ant_split[1]))
            info_all[count_phase_ant].append(float(phase_ant_split[2]))
            info_all[count_phase_ant].append(float(phase_ant_split[3]))
            info_all[count_phase_ant].append(float(phase_ant_split[4]))
            info_all[count_phase_ant].append(float(phase_ant_split[5]))
            info_all[count_phase_ant].append(float(phase_ant_split[6]))
            count_phase_ant += 1
        count += 1

    all_info_all2.append(info_all)
# cholec80==================

'''
with open('./Miccai19.pkl', 'wb') as f:
    pickle.dump(all_info_all, f)

with open('./Miccai19.pkl', 'rb') as f:
    all_info_19 = pickle.load(f)
'''

with open('./cholec80.pkl', 'wb') as f:
    pickle.dump(all_info_all2, f)

with open('./cholec80.pkl', 'rb') as f:
    all_info_80 = pickle.load(f)
'''
# Miccai19==================
train_file_paths_19 = []
train_labels_19 = []
train_num_each_19 = []

val_file_paths_19_1 = []
val_labels_19_1 = []
val_num_each_19_1 = []

val_file_paths_19_2 = []
val_labels_19_2 = []
val_num_each_19_2 = []

for i in range(2, 22):
    train_num_each_19.append(len(all_info_19[i]))
    for j in range(len(all_info_19[i])):
        train_file_paths_19.append(all_info_19[i][j][0])
        train_labels_19.append(all_info_19[i][j][1:])

print(len(train_file_paths_19))
print(len(train_labels_19))

for i in [0, 1]:
    val_num_each_19_1.append(len(all_info_19[i]))
    for j in range(len(all_info_19[i])):
        val_file_paths_19_1.append(all_info_19[i][j][0])
        val_labels_19_1.append(all_info_19[i][j][1:])

print(len(val_file_paths_19_1))
print(len(val_labels_19_1))

for i in [22, 23]:
    val_num_each_19_2.append(len(all_info_19[i]))
    for j in range(len(all_info_19[i])):
        val_file_paths_19_2.append(all_info_19[i][j][0])
        val_labels_19_2.append(all_info_19[i][j][1:])

print(len(val_file_paths_19_2))
print(len(val_labels_19_2))

# Miccai19==================
'''

# cholec80==================
train_file_paths_80 = []
test_file_paths_80 = []
val_file_paths_80 = []
val_labels_80 = []
train_labels_80 = []
test_labels_80 = []

train_num_each_80 = []
val_num_each_80 = []
test_num_each_80 = []

stat = np.zeros(7).astype(int)
for i in range(40):
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

for i in range(40, 48):
    val_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        val_file_paths_80.append(all_info_80[i][j][0])
        val_labels_80.append(all_info_80[i][j][1:])

for i in range(40, 80):
    test_num_each_80.append(len(all_info_80[i]))
    for j in range(len(all_info_80[i])):
        test_file_paths_80.append(all_info_80[i][j][0])
        test_labels_80.append(all_info_80[i][j][1:])

print(len(val_file_paths_80))
print(len(val_labels_80))

# cholec80==================


train_val_test_paths_labels = []
#train_val_test_paths_labels.append(train_file_paths_19)
train_val_test_paths_labels.append(train_file_paths_80)
train_val_test_paths_labels.append(val_file_paths_80)

#train_val_test_paths_labels.append(train_labels_19)
train_val_test_paths_labels.append(train_labels_80)
train_val_test_paths_labels.append(val_labels_80)

#train_val_test_paths_labels.append(train_num_each_19)
train_val_test_paths_labels.append(train_num_each_80)
train_val_test_paths_labels.append(val_num_each_80)


train_val_test_paths_labels.append(test_file_paths_80)
train_val_test_paths_labels.append(test_labels_80)
train_val_test_paths_labels.append(test_num_each_80)


with open('train_val_paths_labels1.pkl', 'wb') as f:
    pickle.dump(train_val_test_paths_labels, f)

print('Done')
print()