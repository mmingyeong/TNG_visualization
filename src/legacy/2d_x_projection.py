import h5py
import dask.array as da
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

import glob
import time
import pynvml
from tqdm import tqdm
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("process.log"),
                        logging.StreamHandler()
                    ])

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
    logging.info(f'GPU Usage. Used: {used} Total: {total} ({percentage:.2f}% used). Free: {free}')

# Replace bin_set_list with a direct assignment
bin_set = 500

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
    hist, _, _ = cp.histogram2d(data[:, 1], data[:, 2], bins=bin_set) # 데이터의 첫 번째와 두 번째 열을 사용하여 히스토그램 계산
    return hist

def process_files(file_paths, x_min, x_max, cache_dir="cache"):
    all_hist_gpu = cp.zeros((500, 500), dtype=cp.float64) # Fix bin_set to 500
    for file_path in tqdm(file_paths):
        schedule_gpu_memory_logging()  # GPU 메모리 사용량 로깅
        cache_path = os.path.join(cache_dir, os.path.basename(file_path).replace('.hdf5', '.npy'))
        
        # 캐시된 데이터 로드
        if os.path.exists(cache_path):
            data = np.load(cache_path)
        else:
            data_dask = load_hdf5_data(file_path)
            data = data_dask.compute() # Dask 데이터를 NumPy 배열로 변환
            np.save(cache_path, data)
            logging.info(f'Cached data for file: {file_path}')
        
        # x_min과 x_max 사이의 데이터만 선택
        mask = (data[:, 0] >= x_min) & (data[:, 0] < x_max)
        data_filtered = data[mask]        
    
        # 필터링된 데이터가 비어 있는지 확인
        if data_filtered.size == 0:
            logging.info(f"No data in x range for file: {file_path}")
            continue
        else:
            data_gpu = cp.array(data_filtered) # NumPy 배열을 CuPy 배열로 변환
            hist_gpu = calculate_hist2d_gpu(data_gpu, 500) # GPU에서 2D 히스토그램 계산 (bin_set = 500)
            all_hist_gpu += hist_gpu # 각 파일의 히스토그램을 전체 히스토그램에 더함

    return all_hist_gpu


# 2D 히스토그램 시각화
def visualize_histogram(hist, bin_set, x_min, x_max):
    plt.clf()  # 이전 그림 지우기
    log_hist = cp.log1p(hist) # 히스토그램의 로그 스케일 적용
    log_hist_np = log_hist.get() # CuPy 배열을 NumPy 배열로 변환
    plt.figure(figsize=(10, 8))
    plt.imshow(log_hist_np, origin='lower', cmap='viridis', vmin=5, vmax=13)
    plt.colorbar(label='log(counts)')
    plt.xlabel('log Y-axis')
    plt.ylabel('log Z-axis')
    plt.title(f'2D Histogram: x-axis projection bin_set={bin_set}, x_range=({x_min},{x_max}) in log scale', pad=20)
    plt.tight_layout()
    plt.savefig(f'2dhist_bin={bin_set}_x_{x_min}-{x_max}_log.pdf', format='pdf')
    plt.savefig(f'2dhist_bin={bin_set}_x_{x_min}-{x_max}_log.png', format='png', dpi=300)
    plt.show()
    plt.close()  # 그림 닫기

# 데이터 캐싱
def cache_data(file_paths, cache_dir="cache"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    for file_path in tqdm(file_paths):
        cache_path = os.path.join(cache_dir, os.path.basename(file_path).replace('.hdf5', '.npy'))
        if not os.path.exists(cache_path):
            data_dask = load_hdf5_data(file_path)
            data = data_dask.compute()  # Dask 데이터를 NumPy 배열로 변환
            np.save(cache_path, data)
            logging.info(f'Cached data for file: {file_path}')
        else:
            logging.info(f'Cache already exists for file: {file_path}, skipping...')

x_min = 0
x_max = 10

if __name__ == "__main__":
    now1 = time.time()
    logging.info("Processing started.")
    # GPU 메모리 사용량 로깅 시작
    schedule_gpu_memory_logging()
    file_paths = glob.glob(f"/home/users/mmingyeong/tng/tng_99_240425/tng_local/snapshot-99.*.hdf5") # HDF5 파일 경로 목록을 수정해야 함
    
    # 데이터 캐싱
    cache_data(file_paths)

    # Fixed x_min and x_max values
    logging.info(f'Processing x range ({x_min}, {x_max}) with bin set 500.')
    all_hist_gpu = process_files(file_paths, x_min, x_max)  # 데이터 처리 함수 호출
        
    cp.savetxt(f'all_hist_gpu_bin_500_x_{x_min}-{x_max}_log.csv', all_hist_gpu, delimiter=',')
    visualize_histogram(all_hist_gpu, 500, x_min, x_max)  # 2D 히스토그램 시각화 (bin_set = 500)

    now2 = time.time()
    logging.info(f"Processing finished. Total time: {now2-now1} seconds")
