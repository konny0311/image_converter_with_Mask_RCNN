import cv2
import numpy as np

THRESH_ROI = 0.1

def make_image_gray(result, img):
    if 1 in result['class_ids']: # 1 is an id of person
        idxs = np.where(result['class_ids'] == 1)[0]
        total_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in idxs:
            if _check_roi(result, img_area, i)
                mask = result['masks'][:,:,i]
                mask_img = _apply_mask(img, mask)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                total_mask += mask_img
        #マスク重なりを修正する
        total_mask = cv2.threshold(total_mask, 200, 255,cv2.THRESH_BINARY)[1]
        total_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        out = np.where(total_mask==np.array([255, 255, 255]), img, gray_img)

        return out

    else:
        return img

def make_image_blur(result, img):
    ori = img.copy()
    img_area = img.shape[0] * img.shape[1]
    if 1 in result['class_ids']: # 1 is an id of person
        idxs = np.where(result['class_ids'] == 1)[0]
        total_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in idxs:
            if _check_roi(result, img_area, i)
                mask = result['masks'][:,:,i]
                mask_img = _apply_mask(img, mask)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                total_mask += mask_img
        #マスク重なりを修正する
        total_mask = cv2.threshold(total_mask, 200, 255,cv2.THRESH_BINARY)[1]
        total_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
        blurred_img = cv2.medianBlur(ori, 9)
        out = np.where(total_mask==np.array([255, 255, 255]), img, blurred_img)

        return out
    
    else:
        return img

def make_image_blur_gray(result, img):
    ori = img.copy()
    img_area = img.shape[0] * img.shape[1]
    if 1 in result['class_ids']: # 1 is an id of person
        idxs = np.where(result['class_ids'] == 1)[0]
        total_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        for i in idxs:
            if _check_roi(result, img_area, i)
                mask = result['masks'][:,:,i]
                mask_img = _apply_mask(img, mask)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                total_mask += mask_img
        #マスク重なりを修正する
        total_mask = cv2.threshold(total_mask, 200, 255,cv2.THRESH_BINARY)[1]
        total_mask = cv2.cvtColor(total_mask, cv2.COLOR_GRAY2BGR)
        blurred_img = cv2.medianBlur(ori, 9)
        blurred_gray_img = _gray_with_3channel(blurred_img)
        out = np.where(total_mask==np.array([255, 255, 255]), img, blurred_gray_img)

        return out
    
    else:
        return img



def _apply_mask(img, mask, color=None, alpha=0.5):
    copied = img.copy()
    for c in range(3):
        if color is None:
            copied[:, :, c] = np.where(mask == 1,
                                    255, 0) # create binary mask
        else:
            copied[:, :, c] = np.where(mask == 1,
                                img[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                img[:, :, c])

    return copied

def _gray_with_3channel(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def _check_roi(result, img_area, i):
    roi = result['rois'][i]
    w = roi[3] - roi[1]
    h = roi[2] - roi[0]
    rate = float((w*h)/img_area)
    if rate > THRESH_ROI:
        return True
    else:
        return False
