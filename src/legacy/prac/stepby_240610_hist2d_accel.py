import h5py
import random
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import pynvml
from pynvml.smi import nvidia_smi
import time
from cupy.cuda import runtime

# 히스토그램 범위 및 빈 설정
bins = 100  # test용으로 일단 10개 설정

# GPU 히스토그램 계산 함수
def calculate_hist2d_gpu(x, y, bins):
    hist2d_res = plt.hist2d(x, y, norm=mpl.colors.LogNorm(), bins=bins)
    #hist2d_res, _, _ = cp.histogram2d(x, y, bins=bins)
    return hist2d_res

# 데이터 처리 함수 (multiprocessing용)
def process_data_gpu(data):
    x_gpu = cp.array(data[:, 0])
    y_gpu = cp.array(data[:, 1])
    # z_gpu = cp.array(data[:, 2])
    hist_res = calculate_hist2d_gpu(x_gpu, y_gpu, bins) # z-axis projection
    # hist = calculate_hist2d_gpu(x_gpu, z_gpu, bins, range_set) # y-axis projection
    # hist = calculate_hist2d_gpu(y_gpu, z_gpu, bins, range_set) # z-axis projection
    return hist_res

# 병렬 처리 결과 수집 함수
def collect_results(result):
    global all_hist_gpu
    all_hist_gpu += result

# 병렬 처리 준비
if __name__ == "__main__":
    now1 = time.time()

    pool = mp.Pool(mp.cpu_count())

    # GPU에서 전체 히스토그램 계산
    all_hist_gpu = cp.zeros((bins, bins), dtype=cp.float64)

    # for num in test_range:
    num = 0
    file_name = f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.{num}.hdf5"
    with h5py.File(file_name, 'r') as f:
        PartType1 = f['PartType1']
        dm_pos = PartType1["Coordinates"][:]
        dm_pos = dm_pos * 0.001  # kpc/h -> Mpc/h 단위 변경
        dm_pos = dm_pos.astype(np.float64)  # 데이터 타입 변경
        
    pool.apply_async(process_data_gpu, args=(dm_pos,), callback=collect_results)

    pool.close()
    pool.join()
    
    print(all_hist_gpu)
    all_hist_cpu = cp.asnumpy(all_hist_gpu)
    print(all_hist_cpu)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(all_hist_cpu.T, origin='lower', aspect='auto', cmap='magma', rasterized=True)
    fig.colorbar(im)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title(f'2D Histogram of 600 Datasets (Combined Optimizations): z-axis projections')

    plt.savefig(f'2d_histogram_rasterized_file{num}.svg', format='svg')
    plt.savefig(f'2d_histogram_rasterized_file{num}.png', format='png')
    plt.show()

    now2 = time.time()
    print(f"process time: {now2-now1}")
