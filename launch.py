import os
import time
import requests
import json


def get_now_time():
    return time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))


def _launch_furion(root_dir, cluster_name, job_name, job_type, docker_path, command, creator, access_key, num_nodes,
                   num_gpus):
    if num_nodes > 1:
        assert num_gpus == 8
    url = 'http://furion.xforwardai.io/apiserver/job/'
    clusters = {
        'sh_idc': '810078b5901a343d553e8564d1979de7',
        'bj_idc': '2c92b280316539d074c7ce27fc815a54',
        'ali_k8s': 'a1e0211a8820fa73369c9ad475ea4a58',
    }
    cpu_num_dic = {
        'bj_idc': 40,
        'sh_idc': 32,
        'ali_k8s': 48,
    }
    gpu_type_dic = {
        'bj_idc': "titan-rtx",
        'sh_idc': 32,
        'ali_k8s': "rtx-2080ti",
    }
    data = {
        'clusterID': clusters[cluster_name],
        'name': job_name,
        'type': job_type,
        'creator': creator,
        'image': docker_path,
        'gpuType': gpu_type_dic[cluster_name],
        'command': command,
        'resources': {
            'worker': {
                'replica': num_nodes,
                'cpu': cpu_num_dic[cluster_name],
                'ram': 64,
                'gpu': num_gpus,
            }
        },
        # 'env': {
        #     # 'PYTHONPATH': '{}:{}/3rdparty:$PYTHONPATH'.format(root_dir, root_dir),
        #     'MXNET_CUDNN_AUTOTUNE_DEFAULT': '0',
        #     'PYTHONUNBUFFERED': '1',
        #     'MXNET_ENABLE_GPU_P2P': '0'
        # }
    }
    headers = {
        'accessKey': access_key,
    }
    r = requests.post(url, json.dumps(data), headers=headers)
    return r


def launch_pytorch(env_command, root_dir, cluster_name, job_name, job_command, creator, access_key, num_nodes=1,
                   num_gpus=8):
    docker_path = 'hub.xforwardai.io/dlp/gait-py36:v4'
    # command = 'cd {} && source /etc/profile && {}'.format(root_dir, job_command)
    command = 'cd {} && {} && {}'.format(root_dir, env_command, job_command)
    print(command)
    _launch_furion(root_dir, cluster_name, job_name, 'pytorchjob', docker_path, command, creator, access_key, num_nodes,
                   num_gpus)


def submit(cfg, copy_root=True):
    root_dir = os.path.abspath(__file__).split('launch.py')[0][:-1]
    time_str = get_now_time()
    if copy_root:
        new_dir_name = root_dir.split('/')[-1]
        if cfg.cluster_name == 'bj_idc':
            new_root_dir = '/mnt/cfs/algorithm/users/xianda.guo/cluster/' + new_dir_name + '-{}'.format(time_str)
            # env_command = 'source /mnt/cfs/algorithm/users/xianda.guo/anaconda3/bin/activate swin'
            env_command = 'source /etc/profile'
        elif cfg.cluster_name == 'ali_k8s':
            new_root_dir = '/mnt/cpfs/users/gpuwork/xianda.guo/cluster/' + new_dir_name + '-{}'.format(time_str)
            # env_command = 'source /mnt/cpfs/users/gpuwork/xianda.guo/data/anaconda3/bin/activate py3env'
            # env_command = 'source /mnt/cpfs/users/gpuwork/xianda.guo/data/anaconda3/bin/activate swin'
            env_command = 'source /etc/profile'
        elif cfg.cluster_name == 'sh_idc':
            new_root_dir = '/mnt/gfs/traincluster/users/xianda.guo/cluster/' + new_dir_name + '-{}'.format(time_str)
        os.system('cp -r {} {}; chmod -R 777 {}'.format(root_dir, new_root_dir, new_root_dir))
        root_dir = new_root_dir
    print(root_dir)
    cfg.job_name += '-{}'.format(time_str)
    launch_pytorch(env_command, root_dir, cfg.cluster_name, cfg.job_name, cfg.job_command, cfg.creator, cfg.access_key,
                   cfg.num_nodes, cfg.num_gpus)


if __name__ == "__main__":
    from config import conf
    from easydict import EasyDict
    print(conf)
    cfg = EasyDict()
    cfg.num_nodes = 1
    cfg.job_command = "python3 train.py --cache=True"
    # cfg.job_command = "python3 train.py --cache=False"
    cfg.creator = "xianda.guo"
    cfg.access_key = "$2a$10$gJQah7ScBmpHgV15DVA9L.GzpV4yE.LZRVzgfrFdaKceH5sOqRybC"
    cfg.num_gpus = 8
    cfg.job_name = 'gaitset-mvlp-grew-master'
    # cfg.cluster_name = "ali_k8s"
    cfg.cluster_name = "bj_idc"

    submit(cfg)