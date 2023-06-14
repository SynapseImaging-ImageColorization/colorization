import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from data_preprocessing import metrics, utils

COLOR_PATH = './dataset/Color'
GRAY_PATH = './dataset/Gray'


def main():
    color_files = utils.get_file_names_in_directory(COLOR_PATH)
    gray_files = utils.get_file_names_in_directory(GRAY_PATH)

    color_files.sort()
    gray_files.sort()

    labels = []
    ssim_results = []
    psnr_results = []
    fid_results = []

    for i in tqdm(range(len(color_files))):
        color = cv2.imread(f'{COLOR_PATH}/{color_files[i]}')
        gray = cv2.imread(f'{GRAY_PATH}/{gray_files[i]}')
        
        color = utils.template_matching(color, gray)

        color = cv2.resize(color, (128, 128), interpolation=cv2.INTER_AREA)

        ssim = metrics.calculate_ssim(color, gray)
        psnr = metrics.calculate_psnr(color, gray)
        fid = metrics.calculate_fid(color, gray)

        labels.append('_'.join(color_files[i].split('_')[:-1]))
        ssim_results.append(ssim)
        psnr_results.append(psnr)
        fid_results.append(fid)

    data = {
        'label': labels,
        'ssim': ssim_results,
        'psnr': psnr_results,
        'fid': fid_results
    } 

    data_df = pd.DataFrame(data)
    data_df.to_csv('./results.csv', index=False)


if __name__=='__main__':
    main()
