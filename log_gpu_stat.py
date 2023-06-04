import time
import psutil
import torch
from torch.utils.tensorboard import SummaryWriter
import py3nvml.py3nvml as nvml
import multiprocessing as mp


def record_gpu_mem(loggername):
    def inner():
        all_gpu_count = 8
        nvml.nvmlInit()
        # handle = nvml.nvmlDeviceGetHandleByIndex(0)
        writer = SummaryWriter(loggername)
        i = 0
        while True:
            for j in range(all_gpu_count):
                handle = nvml.nvmlDeviceGetHandleByIndex(j)
                meminfo = nvml.nvmlDeviceGetMemoryInfo(handle)
                # GB
                gpu_mem = (meminfo.total - meminfo.free) / 1024 / 1024 / 1024
                writer.add_scalar('GPU Memory Usage/{}'.format(j), gpu_mem, global_step=i)
                print('GPU Memory Usage{}:'.format(j), gpu_mem, " {}".format(j))
                time.sleep(1)
            i += 1
            time.sleep(1)
    p = mp.Process(target=inner, args=())
    # p.daemon = True
    p.start()



def record_system_mem(loggername):
    def inner():
        writer = SummaryWriter(loggername)
        i = 0
        while True:
            mem_info = psutil.virtual_memory()
            total_mem = mem_info.total / 1024 / 1024 / 1024
            free_mem = mem_info.available / 1024 / 1024 / 1024
            used_mem = total_mem - free_mem
            writer.add_scalar('System Memory Usage', used_mem, global_step=i)
            writer.add_scalar('System Free Memory', free_mem, global_step=i)
            # print('System Memory Usage:', used_mem, " {}".format(i))
            print('System Free Memory:', free_mem, " {}".format(i))
            i += 1
            time.sleep(1)
    p = mp.Process(target=inner, args=())
    # p.daemon = True
    p.start()

if __name__ == '__main__':
    # 创建 TensorBoard 的写入器对象
    # writer = SummaryWriter('logs')

    # 创建记录 GPU 显存占用情况的子进程
    record_gpu_mem("logs")

    record_system_mem("logs")

    # 在主程序中执行其他任务

    # 结束子进程
    # p.terminate()

    # 等待子进程结束
    while True:
        pass 