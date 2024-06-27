#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: Mingyeong Yang (mmingyeong@kasi.re.kr)
# @Date: 2024-06-10
# @Filename: 240610_hist2d_acceleration.py

import h5py
import random
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import pynvml
from pynvml.smi import nvidia_smi
import time
from cupy.cuda import runtime

# 히스토그램 범위 및 빈 설정
bins = 100 # test용으로 일단 10개 설정
# range = [[0, 200], [0, 200]] # min, max

def schedule_gpu_memory_logging():
    nvsmi = nvidia_smi.getInstance()
    res_memory_list = nvsmi.DeviceQuery("memory.free, memory.total, memory.used")["gpu"]
    res_util_list = nvsmi.DeviceQuery("utilization.gpu, memory.free, memory.total, memory.used")["gpu"]

    total = np.average([each["fb_memory_usage"]["total"] for each in res_memory_list])
    used = np.average([each["fb_memory_usage"]["used"] for each in res_memory_list])
    free = np.average([each["fb_memory_usage"]["free"] for each in res_memory_list])
    percentage = 100*used/total
    utilization = np.average([each["utilization"]["gpu_util"] for each in res_util_list])
    
    print(
        f'GPU Usage. Util: {utilization} Used: {used} Total: {total} ({percentage}% used). Free: {free}'
    )

# GPU 히스토그램 계산 함수
def calculate_hist2d_gpu(x, y, bins, range):
    hist, _, _ = cp.histogram2d(x, y, bins=bins, range=range)
    return hist

# 데이터 처리 함수 (multiprocessing용)
def process_data_gpu(data):
    x_gpu = cp.array(data[:, 0])
    y_gpu = cp.array(data[:, 1])
    z_gpu = cp.array(data[:, 2])
    hist = calculate_hist2d_gpu(x_gpu, y_gpu, bins) # z-axis projection
    # hist = calculate_hist2d_gpu(x_gpu, z_gpu, bins, range) # y-axis projection
    # hist = calculate_hist2d_gpu(y_gpu, z_gpu, bins, range) # z-axis projection
    return hist

# 병렬 처리 결과 수집 함수
def collect_results(result):
    global all_hist_gpu
    all_hist_gpu += result

# HDF5 파일에서 데이터를 읽는 함수
def read_data_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        PartType1 = f['PartType1']
        dm_pos = PartType1["Coordinates"][:]
        dm_pos = dm_pos*0.001 # kpc/h -> Mpc/h 단위 변경
        dm_pos = dm_pos.astype(np.float64) # 데이터 타입 변경
        
    return dm_pos

# HDF5 파일에서 데이터를 읽고 GPU 배열로 변환
# file_name = f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.{num}.hdf5"
dataset_name = 'dm_pos'
#test_range = random.sample(range(1, 600), 5)
test_range = [252,
 386,
 535,
 391,
 187,
 32,
 139,
 408,
 517,
 98,
 168,
 226,
 39,
 508,
 374,
 481,
 279,
 573,
 224,
 285]
len_sample = len(test_range)

# 병렬 처리 준비
if __name__ == "__main__":
    mem_info = runtime.memGetInfo()
    now1 = time.time()
    start_mem = mem_info[0]
    schedule_gpu_memory_logging()
    pool = mp.Pool(mp.cpu_count())

    # GPU에서 전체 히스토그램 계산
    all_hist_gpu = cp.zeros((bins, bins), dtype=cp.float64)

    #for i in range(num_datasets):
    for num in test_range:
        file_name = f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.{num}.hdf5"
        data = read_data_from_hdf5(file_name)
        pool.apply_async(process_data_gpu, args=(data,), callback=collect_results)

    pool.close()
    pool.join()

    # CPU로 데이터 복사
    all_hist_cpu = cp.asnumpy(all_hist_gpu)

    # 최종 히스토그램 계산 및 그리기
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(all_hist_cpu.T, origin='lower', aspect='auto', cmap='viridis', rasterized=True)
    fig.colorbar(im)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(f'2D Histogram of 600 Datasets (Combined Optimizations): z-axis projection bins={bins}, len={len_sample}')

    # 래스터화된 벡터 그래픽으로 저장
    plt.savefig(f'2d_histogram_rasterized_ver1_bins={bins}_len={len_sample}.svg', format='svg')
    plt.show()
    
    now2 = time.time()
    end_mem = runtime.memGetInfo()[0]
    # 데이터 처리 중 GPU에서 사용된 메모리 출력
    gpu_mem_used = end_mem - start_mem
    print(f"process time: {now2-now1}")
    print("GPU Memory Used:", gpu_mem_used / 1024**2, "MB")
    
