import cv2
import numpy as np


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)

    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)

    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def simple_color_balance(input_img, s1, s2):
    h, w = input_img.shape[:2]
    temp_img = input_img.copy()
    one_dim_array = temp_img.flatten()
    sort_array = sorted(one_dim_array)

    per1 = int((h * w) * s1 / 100)
    minvalue = sort_array[per1]

    per2 = int((h * w) * s2 / 100)
    maxvalue = sort_array[(h * w) - 1 - per2]

    # 实施简单白平衡算法
    if (maxvalue <= minvalue):
        out_img = np.full(input_img.shape, maxvalue)
    else:
        scale = 255.0 / (maxvalue - minvalue)
        # out_img = np.where(temp_img < minvalue, 0)    # 防止像素溢出
        # out_img = np.where(out_img > maxvalue, 255)   # 防止像素溢出
        temp_img[temp_img<minvalue] = 0
        temp_img[temp_img>maxvalue] = 255
        out_img = scale * (temp_img - minvalue)        # 映射中间段的图像像素
        out_img = cv2.convertScaleAbs(out_img)
    return out_img

def MSRCR(img, scales, s1, s2):
    h, w = img.shape[:2]
    scles_size = len(scales)
    log_R = np.zeros((h, w), dtype=np.float32)

    img_sum = np.add(img[:,:,0],img[:,:,1],img[:,:,2])
    img_sum = replaceZeroes(img_sum)
    gray_img = []

    for j in range(3):
        img[:, :, j] = replaceZeroes(img[:, :, j])
        for i in range(0, scles_size):
            L_blur = cv2.GaussianBlur(img[:, :, j], (scales[i], scales[i]), 0)
            L_blur = replaceZeroes(L_blur)

            dst_img = cv2.log(img[:, :, j]/255.0)
            dst_Lblur = cv2.log(L_blur/255.0)
            dst_ixl = cv2.multiply(dst_img, dst_Lblur)
            log_R += cv2.subtract(dst_img, dst_ixl)

        MSR = log_R / 3.0
        MSRCR = MSR * (cv2.log(125.0 * img[:, :, j]) - cv2.log(img_sum))
        gray = simple_color_balance(MSRCR, s1, s2)
        gray_img.append(gray)
    return gray_img

def MSRCP(img, scales, s1, s2):
    h, w = img.shape[:2]
    scales_size = len(scales)
    B_chan = img[:, :, 0]
    G_chan = img[:, :, 1]
    R_chan = img[:, :, 2]
    log_R = np.zeros((h, w), dtype=np.float32)
    array_255 = np.full((h, w),255.0,dtype=np.float32)

    I_array = (B_chan + G_chan + R_chan) / 3.0
    I_array = replaceZeroes(I_array)

    for i in range(0, scales_size):
        L_blur = cv2.GaussianBlur(I_array, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_I = cv2.log(I_array/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_ixl = cv2.multiply(dst_I, dst_Lblur)
        log_R += cv2.subtract(dst_I, dst_ixl)
    MSR = log_R / 3.0
    Int1 = simple_color_balance(MSR, s1, s2)

    B_array = np.maximum(B_chan,G_chan,R_chan)
    A = np.minimum(array_255 / B_array, Int1/I_array)
    R_channel_out = A * R_chan
    G_channel_out = A * G_chan
    B_channel_out = A * B_chan

    MSRCP_Out_img = cv2.merge([B_channel_out, G_channel_out, R_channel_out])
    MSRCP_Out = cv2.convertScaleAbs(MSRCP_Out_img)

    return MSRCP_Out

