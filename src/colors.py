import pandas as pd
import matplotlib.pyplot as plt

import cupy as cp

# CSV 파일 불러오기
csv_file_path = '/home/users/mmingyeong/TNG_visualization/240627_2Dplot/all_hist_gpu_bin_500_x_0-10_log.csv'
data = pd.read_csv(csv_file_path)

# 사용할 색상 맵 리스트
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

# 2D 히스토그램 시각화
def visualize_histogram(hist, cmap):
    plt.clf()  # 이전 그림 지우기
    hist_cp = cp.asarray(hist)
    log_hist = cp.log1p(hist_cp) # 히스토그램의 로그 스케일 적용
    log_hist_np = log_hist.get() # CuPy 배열을 NumPy 배열로 변환
    plt.figure(figsize=(10, 8))
    plt.imshow(log_hist_np, origin='lower', cmap=cmap, vmin=5, vmax=13, rasterized=True)
    plt.colorbar(label='log(counts)')
    plt.xlabel('log Y-axis')
    plt.ylabel('log Z-axis')
    plt.title(f'2D Histogram: x-axis projection bin number=500, x_range=(0,10) in log scale', pad=20)
    plt.tight_layout()
    plt.savefig(f'2dhist_bin=500_x_{cmap}.pdf', format='pdf', dpi=600)
    plt.savefig(f'2dhist_bin=500_x_{cmap}.png', format='png', dpi=300)
    plt.show()
    plt.close()  # 그림 닫기

for cmap in cmaps:
    visualize_histogram(data, cmap)  # 2D 히스토그램 시각화 (bin_set = 500)