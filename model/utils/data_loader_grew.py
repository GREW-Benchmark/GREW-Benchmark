import os
import os.path as osp

import numpy as np

from .data_set import DataSet

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


def read_txt(file_list_path):
    f = open(file_list_path, 'r')
    filelist = [l.strip('\n') for l in f.readlines()]
    return filelist


def data_loader_grew(dataset_path, resolution, dataset, pid_num, pid_shuffle, cache=True, listCollection_path=''):
    print('load grew data:')
    # work_path = '/mnt/cpfs/users/gpuwork/xianda.guo/work/'
    # listCollection_path = work_path + 'partition/all_list_grew_iccv2021.npy'

    seq_dir, label, seq_type, view = check_listCollection(dataset_path,listCollection_path)

    train_source = DataSet(seq_dir, label, seq_type, view, cache, resolution)
    test_source = None
    print('len train,test--', len(train_source))
    return train_source, test_source
