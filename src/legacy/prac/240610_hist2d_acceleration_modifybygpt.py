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

# NVML 초기화
pynvml.nvmlInit()

# 히스토그램 범위 및 빈 설정
bins = 100  # test용으로 일단 10개 설정
# range_set = [[0, 200], [0, 200]] # min, max

# GPU 메모리 사용량 로깅 함수
def schedule_gpu_memory_logging():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    res = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = res.total
    used = res.used
    free = res.free
    percentage = 100 * used / total
    print(
        f'GPU Usage. Used: {used} Total: {total} ({percentage:.2f}% used). Free: {free}'
    )

# GPU 히스토그램 계산 함수
def calculate_hist2d_gpu(x, y, bins):
    hist2d_res = plt.hist2d(x, y, norm=mpl.colors.LogNorm(), bins=500)
    #hist2d_res, _, _ = cp.histogram2d(x, y, bins=bins)
    #print(hist2d_res)
    return hist2d_res

# 데이터 처리 함수 (multiprocessing용)
def process_data_gpu(data):
    x_gpu = cp.array(data[:, 0])
    y_gpu = cp.array(data[:, 1])
    z_gpu = cp.array(data[:, 2])
    hist_res = calculate_hist2d_gpu(x_gpu, y_gpu, bins) # z-axis projection
    # hist = calculate_hist2d_gpu(x_gpu, z_gpu, bins, range_set) # y-axis projection
    # hist = calculate_hist2d_gpu(y_gpu, z_gpu, bins, range_set) # z-axis projection
    return hist_res

# 병렬 처리 결과 수집 함수
def collect_results(result):
    global all_hist_gpu
    all_hist_gpu += result

# HDF5 파일에서 데이터를 읽는 함수
def read_data_from_hdf5(file_name):
    with h5py.File(file_name, 'r') as f:
        PartType1 = f['PartType1']
        dm_pos = PartType1["Coordinates"][:]
        dm_pos = dm_pos * 0.001  # kpc/h -> Mpc/h 단위 변경
        dm_pos = dm_pos.astype(np.float64)  # 데이터 타입 변경
    return dm_pos

# 병렬 처리 준비
if __name__ == "__main__":
    now1 = time.time()
    # GPU 메모리 사용량 로깅 시작
    schedule_gpu_memory_logging()
    
    mem_info = runtime.memGetInfo()
    start_mem = mem_info[0]

    pool = mp.Pool(mp.cpu_count())

    # GPU에서 전체 히스토그램 계산
    all_hist_gpu = cp.zeros((bins, bins), dtype=cp.float64)
    # test_range = random.sample(range(1, 600), 10)
    test_range = [1, 10, 100, 300, 500]
    len_sample = len(test_range)

    # for num in test_range:
    num = 432
    file_name = f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.{num}.hdf5"
    data = read_data_from_hdf5(file_name)
    pool.apply_async(process_data_gpu, args=(data,), callback=collect_results)

    pool.close()
    pool.join()

    end_mem = runtime.memGetInfo()[0]

    all_hist_cpu = cp.asnumpy(all_hist_gpu)

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
    print("GPU Memory Used:", (start_mem - end_mem) / 1024 ** 2, "MB")
