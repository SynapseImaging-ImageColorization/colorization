import cv2
import matplotlib.pyplot as plt

from data_preprocessing import metrics, utils

COLOR_PATH = './dataset/Color'
GRAY_PATH = './dataset/Gray'


def main():
    color_files = utils.get_file_names_in_directory(COLOR_PATH)
    gray_files = utils.get_file_names_in_directory(GRAY_PATH)

    ssim_results = []

    for i in range(1000):
        color = cv2.imread(f'{COLOR_PATH}/{color_files[i]}')
        gray = cv2.imread(f'{GRAY_PATH}/{gray_files[i]}')

        color = cv2.resize(color, (128, 128), interpolation=cv2.INTER_AREA)

        ssim = metrics.calculate_ssim(color, gray)
        psnr = metrics.calculate_psnr(color, gray)
        # fid = metrics.calculate_fid(color, gray)

        ssim_results.append((ssim, psnr))

    plt.figure()
    plt.scatter(*zip(*ssim_results))
    plt.xlabel('SSIM')
    plt.ylabel('PSNR')
    plt.show()


if __name__=='__main__':
    main()
