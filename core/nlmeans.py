import numpy as np
import cv2 as cv
import math


def padding(img, w=2):
    X, Y, _ = img.shape
    padded_img = np.zeros((X+2*w, Y+2*w, 3))
    padded_img[w:X+w, w:Y+w, :] = img
    return padded_img


def f(img, x, y, u, v, h=3, w=2):
    X, Y = img.shape
    diff = np.mean(img[x:x+2*w, y:y+2*w]) - np.mean(img[u:u+2*w, v:v+2*w])
    f = math.exp(-diff*diff/h/h)
    return f


def nlmeans_algorithm(img, h=3, w=2):
    X, Y, _ = img.shape
    denoised_img = np.zeros_like(img).astype(float)
    padded_img = padding(img, w)
    padded_img_grey = np.mean(padded_img, axis=2)
    print(padded_img_grey.shape)
    F = np.zeros((X, Y, X, Y))
    C = np.zeros((X, Y))
    # p = (i, j)
    for i in range(X):
        for j in range(Y):
            print(i, j)
            for u in range(X):
                for v in range(Y):
                    fpq = f(padded_img_grey, i, j, u, v, 3, w)
                    F[i, j, u, v] = fpq
                    denoised_img[i, j, :] += img[u, v, :].astype(float)*fpq
    C = np.sum(np.sum(F, axis=3), axis=2)
    denoised_img /= C.unsqueeze(2)
    return denoised_img


if __name__ == '__main__':
    img = cv.imread('data/1.jpg')
    denoised_img = nlmeans_algorithm(img, h=3, w=1)
    cv.imwrite('result/denoised.jpg')
