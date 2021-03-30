import argparse, cv2, utils
import numpy as np
import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(description="Implementation of Image Processing 2020")
    
    parser.add_argument('--img_dir', 
            type=str, 
            default='../Images',
            help='The directory where the image files are located',
            required=False)
    
    return parser.parse_args()

def gauss_conv():
    """ week3 assignment 1: gaussian filter convolution"""
    
    args = get_arguments()
    ip = utils.ImageProcessing(args)
    
    # load a image
    image = ip.get_one_image()

    # get intensity
    img_cvt = ip.cvtYCrCb(image)
    old_intensity = img_cvt[:, :, 0]
    cv2.imshow('original_intensity', old_intensity)

    intensity = np.concatenate(old_intensity, axis=0)
    # get histogram of the old image
    hist_old = get_histogram(intensity)
    # create transform function vector
    tf_vec = 255. * np.cumsum(hist_old)
    # apply transform function vector
    new_intensity = np.take(tf_vec, intensity)
    new_intensity = new_intensity.astype(np.uint8)
    # get histogram of the new image
    hist_new = get_histogram(new_intensity)
    # reshape according to the old image
    new_intensity = new_intensity.reshape(old_intensity.shape)
    cv2.imshow('equalized_intensity', new_intensity)

    # apply the new intensity to the image
    img_cvt[:, :, 0] = new_intensity
    img_bgr = cv2.cvtColor(img_cvt, cv2.COLOR_YCrCb2BGR)
    cv2.imshow('transformed image', img_bgr)

    fig, ax = plt.subplots(2)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax[0].plot(hist_old)
    ax[0].set_title("Histogram before equalization")
    ax[1].plot(hist_new)
    ax[1].set_title("Histogram after equalization")
    plt.show()



if __name__ == '__main__':
    hist_equal()