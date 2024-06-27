import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import pandas as pd

bin_set = 50

# CSV 파일을 읽는 함수
def read_csv_file(file_path):
    data = pd.read_csv(file_path)
    np_data = data.values
    cp_data = cp.array(np_data)
    return cp_data

file_path = 'all_hist_gpu.csv'  # 읽고자 하는 CSV 파일 경로
hist = read_csv_file(file_path)

log_hist = cp.log1p(hist) # 히스토그램의 로그 스케일 적용
log_hist_np = log_hist.get() # CuPy 배열을 NumPy 배열로 변환
plt.imshow(log_hist_np, origin='lower', cmap='viridis')
plt.colorbar(label='log(counts)')
plt.xlabel('log X-axis')
plt.ylabel('log Y-axis')
plt.title(f'2D Histogram: z-axis projection bin_set={bin_set}')
plt.savefig('histogram.png', format='png', dpi=300)
plt.savefig('histogram.pdf', format='pdf')
plt.show()