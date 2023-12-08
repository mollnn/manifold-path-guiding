import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from matplotlib import pyplot as plt
import cv2
import numpy as np

def imcrop(a, crop = {}):
    x = 0 if "x" not in crop.keys() else crop["x"]
    y = 0 if "y" not in crop.keys() else crop["y"]
    w = a.shape[1] if "w" not in crop.keys() else crop["w"]
    h = a.shape[0] if "h" not in crop.keys() else crop["h"]
    return a[y:y+h, x:x+w, :]

def readexr(a, crop = {}):
    imga = cv2.imread(a, cv2.IMREAD_UNCHANGED).astype("float")[:, :, [2, 1, 0]]
    return imcrop(imga, crop)

def rmse(imageA, imageB, vmax = 1e3):
    imageA = np.minimum(imageA, vmax)
    imageB = np.minimum(imageB, vmax)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err ** 0.5

def mse(imageA, imageB, vmax = 1e3):
    imageA = np.minimum(imageA, vmax)
    imageB = np.minimum(imageB, vmax)
    imageA = np.maximum(imageA, 0)
    imageB = np.maximum(imageB, 0)
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def mae(imageA, imageB):
    err = np.sum(((imageA.astype("float") - imageB.astype("float")) ** 2) ** 0.5)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def mape(imageA, imageB):
    err = np.sum(((imageA.astype("float") - imageB.astype("float")) ** 2) ** 0.5 / (imageB.astype("float")+ 1e-2))
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def relmse(imageA, imageB):
    mat = ((imageA.astype("float") - imageB.astype("float")) ** 2) / (imageB.astype("float") ** 2 + 1e-2)
    data = mat.reshape(-1)
    # Discard the highest 0.1% [Ruppert et al. 2020]
    max_val = np.percentile(data, 99.9)
    max_val = 1e9
    filtered_data = data[data <= max_val]
    err = np.sum(data)
    err /= float(imageA.shape[0] * imageA.shape[1])
    err /= 3
    return err

def gamma_correction(img):
    return np.power(img, 1.0 / 2.2)

def draw(test_id, timeout_list, output_name_list, gt_filename, vmax=1e3):
    if gt_filename != "":
        img_ref = readexr(gt_filename)
    mse_list = []
    for i, output_name in enumerate(output_name_list):
        plt.subplot(len(timeout_list), len(output_name_list) // len(timeout_list), i + 1)
        plt.axis('off')
        img = readexr(f"results/{test_id}/{output_name}.exr")
        plt.imshow(gamma_correction(img))
        if gt_filename != "":
            mse_val = mse(img, img_ref, vmax)
            plt.title(f"{output_name}\nMSE={mse_val:.4f}")
            mse_list.append(mse_val)
    mse_list = np.array(mse_list).reshape(len(timeout_list), -1).T
    if gt_filename != "":
        plt.figure(figsize=(5,4))
        for i, x in enumerate(mse_list):
            plt.plot(timeout_list, x, label=output_name_list[i])
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
    plt.show()
