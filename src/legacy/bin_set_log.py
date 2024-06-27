import h5py
import dask.array as da
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import glob
import time
import pynvml
from tqdm import tqdm

# NVML 초기화
pynvml.nvmlInit()

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

bin_set_list = [250, 125, 100, 50]

# Dask를 사용하여 HDF5 파일에서 데이터 로드
def load_hdf5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        PartType1 = f['PartType1']
        dm_pos = PartType1["Coordinates"][:]
        dm_pos = dm_pos * 0.001  # kpc/h -> Mpc/h 단위 변경
        dm_pos = dm_pos.astype(np.float64)  # 데이터 타입 변경
        data_dask = da.from_array(dm_pos, chunks='auto')
    return data_dask

# GPU에서 2D 히스토그램 계산
def calculate_hist2d_gpu(data, bin_set):
    hist, _, _ = cp.histogram2d(data[:, 0], data[:, 1], bins=bin_set) # 데이터의 첫 번째와 두 번째 열을 사용하여 히스토그램 계산
    return hist

# 모든 파일의 데이터를 로드하고 연산 수행
def process_files(file_paths, z_min, z_max, bin_set):
    all_hist_gpu = cp.zeros((bin_set, bin_set), dtype=cp.float64) # GPU에서 사용할 총 히스토그램 초기화
    for file_path in tqdm(file_paths):
        schedule_gpu_memory_logging()  # GPU 메모리 사용량 로깅
        data_dask = load_hdf5_data(file_path)
        data = data_dask.compute() # Dask 데이터를 NumPy 배열로 변환
        
        # z_min과 z_max 사이의 데이터만 선택
        mask = (data[:, 2] >= z_min) & (data[:, 2] <= z_max)
        data_filtered = data[mask]        
    
        # 필터링된 데이터가 비어 있는지 확인
        if data_filtered.size == 0:
            print(f"No data in z range for file: {file_path}")
            continue
        else:
            data_gpu = cp.array(data_filtered) # NumPy 배열을 CuPy 배열로 변환
            hist_gpu = calculate_hist2d_gpu(data_gpu, bin_set) # GPU에서 2D 히스토그램 계산
            all_hist_gpu += hist_gpu # 각 파일의 히스토그램을 전체 히스토그램에 더함

    return all_hist_gpu

# 2D 히스토그램 시각화
def visualize_histogram(hist, bin_set):
    log_hist = cp.log1p(hist) # 히스토그램의 로그 스케일 적용
    log_hist_np = log_hist.get() # CuPy 배열을 NumPy 배열로 변환
    plt.imshow(log_hist_np, origin='lower', cmap='viridis')
    plt.colorbar(label='log(counts)')
    plt.xlabel('log X-axis')
    plt.ylabel('log Y-axis')
    plt.title(f'2D Histogram: z-axis projection bin_set={bin_set} in log scale')
    plt.savefig(f'2dhist_bin={bin_set}_log.pdf', format='pdf')
    plt.savefig(f'2dhist_bin={bin_set}_log.png', format='png', dpi=300)
    plt.show()

if __name__ == "__main__":
    now1 = time.time()
    # GPU 메모리 사용량 로깅 시작
    schedule_gpu_memory_logging()
    file_paths = glob.glob(f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.*.hdf5") # HDF5 파일 경로 목록을 수정해야 함
    z_min, z_max = 100, 105
    
    for bin_set in bin_set_list:
        all_hist_gpu = process_files(file_paths, z_min, z_max, bin_set) # 모든 파일의 데이터를 처리하여 GPU에서 히스토그램 계산
        cp.savetxt(f'all_hist_gpu_bin_{bin_set}_log.csv', all_hist_gpu, delimiter=',')
        visualize_histogram(all_hist_gpu, bin_set) # 2D 히스토그램 시각화
    
    now2 = time.time()
    print(f"process time: {now2-now1}")
