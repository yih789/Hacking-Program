import cv2
import numpy as np
import re

BLUE = 0
GREEN = 1
RED = 2

# 특정한 색상의 모든 단어가 포함된 이미지를 추출한다.
def get_chars(image, color):
    other_1 = (color + 1) % 3
    other_2 = (color + 2) % 3

    # ex) color=B일 때
    # 순수 G인 경우 검은색으로 변경
    c = image[:, :, other_1] == 255
    image[c] = [0, 0, 0]
    # 순수 R인 경우 검은색으로 변경
    c = image[:, :, other_2] == 255
    image[c] = [0, 0, 0]
    # R과 G가 섞인 부분 제거(AA)
    c = image[:, :, color] < 170 
    image[c] = [0, 0, 0]
    # 순수 B 또는 B가 섞인 부분만 남게 되며 흰색으로 변경
    c = image[:, :, color] != 0
    image[c] = [255, 255, 255]

    return image

# 전체 이미지에서 왼쪽부터 단어별로 이미지를 추출합니다.
def extract_chars(image):
    chars = []
    colors = [BLUE, GREEN, RED]
    for color in colors:
        image_from_one_color = get_chars(image.copy(), color)
        image_gray = cv2.cvtColor(image_from_one_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
        # RETR_EXTERNAL 옵션으로  숫자의 외각을 기준으로 분리
        i, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 추출된 이미지 크기가 50 이상인 경우만 실제 문자 데이터인 것으로 파악
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, width, height = cv2.boundingRect(contour)
                roi = image_gray[y:y+height, x:x+width]
                chars.append((x, roi))

    # x축 기준 왼쪽부터 데이터를 정렬
    chars = sorted(chars, key=lambda char: char[0])
    return chars

# 특정 이미지를 (20 x 20) 크기로 Scaling
def resize20(image):
    # 추출된 이미지들의 크기가 모두 다르므로
    resized = cv2.resize(image, (20, 20))
    # 학습모델은 일자의 형태로 반환 해야한다.
    return resized.reshape(-1, 400).astype(np.float32)

def remove_first_0(string):
    temp = []
    for i in string:
        if i =='+' or i=='-' or i=='*':
            temp.append(i)
    split = re.split('\*|\+|-', string)
    i = 0
    temp_count = 0
    result = ""
    for a in split:
        a = a.lstrip('0')
        if a == '':
            a = '0'
        result += a
        if i < len(split) - 1:
            result += temp[temp_count]
            temp_count = temp_count + 1
        i = i+1
    return result