"""
Name: Phú Minh Nhật
Class: K63K2
MSSV: 18020976

You should understand the code you write.
"""

import numpy as np
import cv2
import argparse


def q_0(input_file, output_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    cv2.imshow('Test img', img)
    cv2.waitKey(5000)

    cv2.imwrite(output_file, img)


def q_1(input_file, output_file):
    """
    Convert the image to gray channel of the input image.
    """
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)

    # Convert image to gray channel
    np_img = np.array(img)
    b = np_img[:,:,0]
    g = np_img[:,:,1]
    r = np_img[:,:,2]
    img_gray = 0.21 * b + 0.72 * g + 0.07 * r
    img_gray = np.array(img_gray, dtype='uint8')
    cv2.imwrite(output_file, img_gray)
    print(np_img)

# luu so lan mau xuat hien cua tung pixel anh
def count(img):
    H=np.zeros(shape=(256,1))
    w,h=img.shape
    for i in range(w):
        for j in range(h):
            k=img[i,j]
            H[k,0]=H[k,0]+1
    return H

def q_2(input_file, output_file):
    """
    Performs a histogram equalization on the input image.
    """
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    
    # Convert image to gray channel
    np_img = np.array(img)
    b = np_img[:,:,0]
    g = np_img[:,:,1]
    r = np_img[:,:,2]
    img_gray = 0.21 * b + 0.72 * g + 0.07 * r
    img_gray = np.array(img_gray, dtype='uint8')
    # Histogram equalization
    w,h=img_gray.shape
    H=count(img_gray)
    y=np.array([])
    # sap xep lai mang theo thu tu tu 0-255
    x=H.reshape(1,256)
    y=np.append(y,x[0,0])
    # T[i]=[i-1]+h[i]
    for i in range(255):
        k=x[0,i+1]+y[i]
        y=np.append(y,k)
    # chia theo cong thuc
    y=np.round(y/(w*h)*255)
    for i in range(w):
        for j in range(h):
            k=img_gray[i,j]
            img_gray[i,j]=y[k]
    cv2.imwrite(output_file, img_gray)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    parser.add_argument("--question", "-q", type=int, default=0, help="Question number")

    args = parser.parse_args()

    q_number = args.question

    if q_number == 1:
        q_1(input_file=args.input_file, output_file=args.output_file)
    elif q_number == 2:
        q_2(input_file=args.input_file, output_file=args.output_file)
    else:
        q_0(input_file=args.input_file, output_file=args.output_file)