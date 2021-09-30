import csv
import os
import os.path as osp
import random

import argparse
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tordata
import xarray as xr
from torch.autograd import Variable
from tqdm import tqdm

from model.network.gaitset import SetNet
from model.utils.data_set import DataSet
import torch.autograd as autograd
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
IF_LOG = True
LOG_PATH = './pretreatment.log'
WORKERS = 0

START = "START"
FINISH = "FINISH"
WARNING = "WARNING"
FAIL = "FAIL"

T_H = 64
T_W = 64


def boolean_string(s):
    if s.upper() not in {'FALSE', 'TRUE'}:
        raise ValueError('Not a valid boolean string')
    return s.upper() == 'TRUE'


def log2str(pid, comment, logs):
    str_log = ''
    if type(logs) is str:
        logs = [logs]
    for log in logs:
        str_log += "# JOB %d : --%s-- %s\n" % (
            pid, comment, log)
    return str_log


def log_print(pid, comment, logs):
    str_log = log2str(pid, comment, logs)
    if comment in [WARNING, FAIL]:
        with open(LOG_PATH, 'a') as log_f:
            log_f.write(str_log)
    if comment in [START, FINISH]:
        if pid % 500 != 0:
            return
    print(str_log, end='')


def img2xarray(vid_path, resolution):
    imgs = sorted(list(os.listdir(vid_path)))
    frame_list = [np.reshape(
        cv2.imread(osp.join(vid_path, _img_path)),
        [resolution, resolution, -1])[:, :, 0]
                  for _img_path in imgs
                  if osp.isfile(osp.join(vid_path, _img_path)) and _img_path.endswith('.png')]
    num_list = list(range(len(frame_list)))
    data_dict = xr.DataArray(
        frame_list,
        coords={'frame': num_list},
        dims=['frame', 'img_y', 'img_x'],
    )
    return data_dict


def collate_fn(batch):
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]
    batch = [seqs, view, seq_type, label, None]
    factor = 1

    sample_type = 'all'
    frame_num = 30

    def Order_select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            if len(frame_set) > frame_num - 1:
                choiceframe_set = frame_set[:len(frame_set) - frame_num + 1]
                frame_id_list = random.choices(choiceframe_set, k=1)
                for i in range(frame_num - 1):
                    frame_id_list.append(frame_id_list[0] + (i + 1))
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                frame_id_list = [0]
                for i in range(len(frame_set) - 1):
                    frame_id_list.append(frame_id_list[0] + (i + 1))
                len_frame_id_list = len(frame_id_list)
                for ll in range(frame_num - len_frame_id_list):
                    frame_id_list.append(frame_id_list[ll])
                _ = [feature.loc[frame_id_list].values for feature in sample]
        else:
            _ = [feature.values for feature in sample]
        return _

    seqs1 = []
    count = 1
    for sq in range(len(seqs)):
        seqs1.append(Order_select_frame(sq))
        count += 1
    seqs = seqs1
    if sample_type == 'random':
        seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    else:
        # print('--2--')
        gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            if i < batch_size
        ] for _ in range(gpu_num)]
        # print(gpu_num,batch_per_gpu,batch_frames)
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
        seqs = [[
            np.concatenate([
                seqs[i][j]
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
        seqs = [np.asarray([
            np.pad(seqs[j][_],
                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                   'constant',
                   constant_values=0)
            for _ in range(gpu_num)])
            for j in range(feature_num)]
        batch[4] = np.asarray(batch_frames)

    batch[0] = seqs
    return batch


def collate_fn_for_probe(batch):
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    seq_type = [batch[i][2] for i in range(batch_size)]
    batch = [seqs, seq_type, None]

    sample_type = 'all'
    frame_num = 30

    def Order_select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            if len(frame_set) > frame_num - 1:
                choiceframe_set = frame_set[:len(frame_set) - frame_num + 1]
                frame_id_list = random.choices(choiceframe_set, k=1)
                for i in range(frame_num - 1):
                    frame_id_list.append(frame_id_list[0] + (i + 1))
                _ = [feature.loc[frame_id_list].values for feature in sample]
            else:
                frame_id_list = [0]
                for i in range(len(frame_set) - 1):
                    frame_id_list.append(frame_id_list[0] + (i + 1))
                len_frame_id_list = len(frame_id_list)
                for ll in range(frame_num - len_frame_id_list):
                    frame_id_list.append(frame_id_list[ll])
                _ = [feature.loc[frame_id_list].values for feature in sample]
        else:
            _ = [feature.values for feature in sample]
        return _

    seqs1 = []
    count = 1
    for sq in range(len(seqs)):
        seqs1.append(Order_select_frame(sq))
        count += 1
    seqs = seqs1
    if sample_type == 'random':
        seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    else:
        # print('--2--')
        gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            if i < batch_size
        ] for _ in range(gpu_num)]
        # print(gpu_num,batch_per_gpu,batch_frames)
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])
        seqs = [[
            np.concatenate([
                seqs[i][j]
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
        seqs = [np.asarray([
            np.pad(seqs[j][_],
                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                   'constant',
                   constant_values=0)
            for _ in range(gpu_num)])
            for j in range(feature_num)]
        batch[2] = np.asarray(batch_frames)

    batch[0] = seqs
    return batch


class DataSet_for_probe(tordata.Dataset):
    def __init__(self, seq_dir, seq_type, cache, resolution , cut=False):
        self.seq_dir = seq_dir
        self.seq_type = seq_type
        self.cache = cache
        self.data_size = len(self.seq_type)
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution)/64*10)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size
        self.cut =cut
        self.seq_type_set = set(self.seq_type)
        _ = np.zeros((len(self.seq_type_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'seq_type': sorted(list(self.seq_type_set)),},
            dims=['seq_type'])
        # print(self.index_dict.shape)
        for i in range(self.data_size):
            _seq_type = self.seq_type[i]
            self.index_dict.loc[_seq_type] = i
        # print(self.index_dict)

    def load_all_data(self):
        # print(self.cache)
        for i in range(self.data_size):
            if i % 10000 ==0:
                print('number-',i)
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        if self.cut:
            return self.img2xarray(
                path)[:, :, self.cut_padding:-self.cut_padding].astype(
                'float32') / 255.0
            # return self.img2xarray(
            #     path)[:, :, self.cut_padding:-self.cut_padding].astype(
            #     'float32')
        else:
            a = self.img2xarray(
                path).astype('float32') / 255.0
            # a = self.img2xarray(
            #     path).astype('float32')
            # print(a.shape)
            return a

    def __getitem__(self, index):
        # pose sequence sampling
        # print(self.cache)
        if not self.cache:
            # print('-1-')
            # print(self.seq_dir[index])
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            # print('-2-')
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            # print('-3-')
            data = self.data[index]
            frame_set = self.frame_set[index]
        return data, frame_set, self.seq_type[index]

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path)) and _img_path.endswith('.png')]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __len__(self):
        return len(self.seq_type)

def check_listCollection(dataset_path,listCollection_path):
    seq_dir = list()
    view = list()
    seq_type = list()
    label = list()

    flg_walk = True
    if os.path.exists(listCollection_path):
        print('    %s is exits.' % listCollection_path)
        all_list = np.load(listCollection_path, allow_pickle=True)
        __seq_dir, _, _, _ = all_list
        print('    check datasetpath and listCollection_name', dataset_path, __seq_dir[0][0])
        if dataset_path in __seq_dir[0][0]:
            seq_dir, label, seq_type, view = all_list
            flg_walk = False

    if flg_walk:
        print('    walk the path:%s' % dataset_path)
        for i, _label in enumerate(sorted(list(os.listdir(dataset_path)))):
            if i % 5000 == 0:
                print('    ', _label)
            label_path = osp.join(dataset_path, _label)
            if os.path.isfile(label_path):
                continue
            for _seq_type in sorted(list(os.listdir(label_path))):
                seq_path = osp.join(label_path, _seq_type)

                # 过滤*_gei.png
                if seq_path.endswith('.png'):
                    continue

                files = os.listdir(seq_path)
                if len(files) > 0:
                    seq_dir.append([seq_path])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append('000')
        print('label length', len(label)), seq_dir
        assert len(label) != 0
        all_list = [seq_dir, label, seq_type, view]
        np.save(listCollection_path, all_list)
    return seq_dir, label, seq_type, view


def check_listCollection_probe(dataset_path,listCollection_path):
    seq_dir = list()
    seq_type = list()

    flg_walk = True
    if os.path.exists(listCollection_path):
        print('    %s is exits.' % listCollection_path)
        all_list = np.load(listCollection_path, allow_pickle=True)
        __seq_dir, _= all_list
        print('    check datasetpath and listCollection_name', dataset_path, __seq_dir[0][0])
        if dataset_path in __seq_dir[0][0]:
            seq_dir, seq_type = all_list
            flg_walk = False

    if flg_walk:
        print('    walk the path:%s' % dataset_path)
        for _seq_type in sorted(list(os.listdir(dataset_path))):
            seq_path = os.path.join(dataset_path, _seq_type)

            # 过滤*_gei.png
            if seq_path.endswith('.png') or os.path.isfile(seq_path):
                continue

            files = os.listdir(seq_path)
            if len(files) > 0:
                seq_dir.append([seq_path])
                seq_type.append(_seq_type)
        print('label length'), seq_dir
        all_list = [seq_dir, seq_type]
        np.save(listCollection_path, all_list)
    return seq_dir, seq_type


def ts2var(x):
    return autograd.Variable(x).cuda()

def np2var(x):
    return ts2var(torch.from_numpy(x))

def extract_test_gallery(test_source, net, gallery_feature_path='./gallery_feature.npy',
                         labelnp_path='./labelnp_l2_48.npy'):
    print('----extract_test_gallery-----')

    data_loader = tordata.DataLoader(
        dataset=test_source,
        batch_size=8,
        sampler=tordata.sampler.SequentialSampler(test_source),
        collate_fn=collate_fn,
        # pin_memory=True,
        num_workers=32)

    feature_list = list()
    label_list = list()
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            seq, view, seq_type, label, batch_frame = x
            if i%100==0:
                print('-->extract_probe:{}'.format(i))
                # print('outputs', outputs.shape())

            for j in range(len(seq)):
                seq[j] =np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame =np2var(batch_frame).int()

            outputs, _ = net(*seq, batch_frame)

            n, _, _ = outputs.size()
            outputs = outputs.view(n, -1).data.cpu().numpy()

            feature_list.append(outputs)
            label_list += label


    # feature = torch.cat(feature_list,0)
    gallery_feature = np.concatenate(feature_list, 0)

    print('GL gallerynp shape = ', gallery_feature.shape)
    np.save(gallery_feature_path, gallery_feature)

    labelnp = np.array(label)
    print('GL labelnp shape = ', labelnp.shape)
    np.save(labelnp_path, labelnp)
    return label_list, gallery_feature


def extract_test_probe(test_source, net,probe_feature_path = './gallery_feature.npy'):
    print('----extract_test_probe-----')

    data_loader = tordata.DataLoader(
        dataset=test_source,
        batch_size=8,
        sampler=tordata.sampler.SequentialSampler(test_source),
        collate_fn=collate_fn_for_probe,
        # pin_memory=True,
        num_workers=16)

    feature_list = list()
    seq_type_list = list()
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            seq, seq_type, batch_frame = x
            for j in range(len(seq)):
                seq[j] =np2var(seq[j]).float()
            if batch_frame is not None:
                batch_frame =np2var(batch_frame).int()

            outputs, _ = net(*seq, batch_frame)

            n, _, _ = outputs.size()
            outputs = outputs.view(n, -1).data.cpu().numpy()

            feature_list.append(outputs)
            seq_type_list += seq_type

            if i%100==0:
                print('-->extract_probe:{}'.format(i))
    # feature = torch.cat(feature_list,0)
    probe_feature = np.concatenate(feature_list, 0)

    print('GL gallerynp shape = ', probe_feature.shape)
    np.save(probe_feature_path, probe_feature)

    return probe_feature,seq_type_list


def cuda_dist(x, y):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    dist = torch.sum(x ** 2, 1).unsqueeze(1) + torch.sum(y ** 2, 1).unsqueeze(
        1).transpose(0, 1) - 2 * torch.matmul(x, y.transpose(0, 1))
    dist = torch.sqrt(F.relu(dist))
    return dist


def build_submission_csv(submission_path, model_path,gallery_path, probe_path,prepare_to_submit_csv_path,
            listCollection_gallery_path='', listCollection_probe_path=''):
    rank = 20

    # 1. read submission.csv
    with open(submission_path, 'r') as f:
        reader = csv.reader(f)
        listcsv = []
        for i, row in enumerate(reader):
            if i == 0:
                print(row)
            listcsv.append(row)
    print('finish reading csv!')

    # build model
    net = SetNet(hidden_dim=256)
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(torch.load(model_path))

    # 2. prepare gallery
    # work_path = '/mnt/cfs/algorithm/users/xianda.guo/work/'
    # listCollection_gallery_path = work_path + 'partition/all_list_grew_test_gallery_iccv2021.npy'
    seq_dir, label, seq_type, view = check_listCollection(gallery_path, listCollection_path=listCollection_gallery_path)
    print(len(seq_dir),len(label),len(seq_type))
    test_source_gallery = DataSet(seq_dir, label, seq_type, view, cache=False, resolution=64)
    label, feature_gallery = extract_test_gallery(test_source_gallery, net)
    print('feature_gallery shape = ', feature_gallery.shape)


    # 3. prepare probe
    # work_path = '/mnt/cfs/algorithm/users/xianda.guo/work/'
    # listCollection_probe_path = work_path + 'partition/all_list_grew_test_probe_iccv2021.npy'
    seq_dir_probe, seq_type_probe = check_listCollection_probe(probe_path,
                                                               listCollection_path=listCollection_probe_path)
    print(len(seq_dir_probe), len(seq_type_probe))
    test_source_probe = DataSet_for_probe(seq_dir_probe, seq_type_probe, cache=False, resolution=64,
                                          cut=True)
    feature_probe, seq_type_list = extract_test_probe(test_source_probe, net)
    print('feature_probe shape = ', feature_probe.shape)

    dist = cuda_dist(feature_probe, feature_gallery)
    idx = dist.sort(1)[1].cpu().numpy()

    for i, vidId in enumerate(seq_type_list):
        for j, _idx in enumerate(idx[i][:rank]):
            listcsv[i][0] = vidId
            listcsv[i][j + 1] = int(label[_idx])

    with open(prepare_to_submit_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in listcsv:
                writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission_path', type=str, help='provide the csv path of the data',
                        default='your_path/submission.csv')
    parser.add_argument('--model_path', type=str, help='provide the model path ',
                        default='your_model_path/checkpoint/*.ptm')
    parser.add_argument('--gallery_path', type=str,
                        default='your_dataset_path/grew/mask_pose/test/gallery')
    parser.add_argument('--probe_path', type=str,
                        default='your_dataset_path/grew/mask_pose/test/probe')
    parser.add_argument('--prepare_to_submit_csv_path', type=str,
                        help='provide the root path of the data',
                        default='submit/submission.csv')

    opt = parser.parse_args()
    build_submission_csv(opt.submission_path, opt.model_path,opt.gallery_path,
                         opt.probe_path,opt.prepare_to_submit_csv_path)
