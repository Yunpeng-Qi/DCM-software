# -*- coding: utf-8 -*-

import os
import subprocess
from functools import partial
from multiprocessing import Pool, Manager, Array

os.chdir("../../")


def start_process(process_info, wait=False, env=None):
    '''
    Start a process

    Args:
      process_info: the name and arguments of the executeble, in a list
      wait: wait until the process is completed

    Return:
      if wait, returns the error code. Otherwise, return the proc object
    '''
    if wait:
        proc = subprocess.Popen(map(str, process_info),
                                stderr=subprocess.STDOUT,
                                stdout=subprocess.PIPE,
                                env=env)
        while True:
            line = proc.stdout.readline()
            print(line.decode('utf-8'), end='')
            if not line: break
        # proc.wait()
        outs, errs = proc.communicate()  # no timeout is set
        if proc.returncode != 0:
            print(outs.decode('utf-8'))
        return proc.returncode
    else:
        proc = subprocess.Popen(map(str, process_info), env=env)
        return proc


def foo(dev_dict, lock, process_info):  # usage, info):
    # select a device with the minimal usage
    usage_min = 1E5  # large enough number
    avail_dev = -1
    lock.acquire()
    for dev, usage in dev_dict.items():
        if usage < usage_min:
            avail_dev = dev
            usage_min = usage
    dev_dict[avail_dev] += 1
    lock.release()

    print(f'processing on dev {avail_dev} usage {usage_min} ', process_info)
    proc_env = os.environ.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = str(avail_dev)
    err = start_process(process_info, wait=True, env=proc_env)
    lock.acquire()
    dev_dict[avail_dev] -= 1
    lock.release()
    return err


class GPUPool:
    def __init__(self, device_ids, proc_per_dev=1):
        self.proc_per_dev = proc_per_dev
        self.manager = Manager()
        self.dev_dict = self.manager.dict()
        self.lock = self.manager.Lock()
        for dev in device_ids:
            self.dev_dict[dev] = 0

    def process_tasks(self, tasks):
        p = Pool(len(self.dev_dict) * self.proc_per_dev)
        print(len(self.dev_dict) * self.proc_per_dev)
        ret = p.map(partial(foo, self.dev_dict, self.lock), tasks)
        return ret


def main():
    dataset_name = "openimages"  # openimages or coco
    dataset_dir = "/Your/Path/openimages/openimages-val-dataset"  # dataset_dir

    dtmV_encode_quality = [1, 2, 3, 4]  # rate point
    tasks = []
    for dtmV_encode_quality in dtmV_encode_quality:
        output_bitstream_dir_path = "./dtmV_output/bitstream/{}/{}".format(
            dataset_name, dtmV_encode_quality)
        log_save_path = "./logs/default/Encoder/{}/{}".format(
            dataset_name, dtmV_encode_quality)
        if not os.path.exists(output_bitstream_dir_path):
            os.makedirs(output_bitstream_dir_path)
        if not os.path.exists(log_save_path):
            os.makedirs(log_save_path)

        for image in os.listdir(dataset_dir):
            img_name = image.split(".")[0]
            input_img_path = os.path.join(dataset_dir, image)
            output_bitstream_path = os.path.join(output_bitstream_dir_path, "{}.bin".format(img_name))
            dtmV_encode_command = ['python', './tools/main.py',
                                   '--ss-enabled-flag', '--ss-w-bboxes-flag',
                                   '--mode', 'compress',
                                   '--input-files-path', input_img_path,
                                   '--byte-stream-path', output_bitstream_path,
                                   '--save', log_save_path,
                                   '--quality', dtmV_encode_quality,
                                   ]
            tasks.append(dtmV_encode_command)

    print(len(tasks))
    ################################################
    # encoding
    cuda_devices = [0, 1]
    processes_per_gpu = 5
    rpool = GPUPool(device_ids=cuda_devices, proc_per_dev=processes_per_gpu)
    err = rpool.process_tasks(tasks)
    print('\n\n=================================================')
    print(f'Encoding error code: {err}')
    print('\n\n')


if __name__ == '__main__':
    main()
