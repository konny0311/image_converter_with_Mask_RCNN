# -*- coding: utf-8 -*-
"""
This is an API server for "splash of color" using Mask R-CNN.
The server recieves an image and turns non-human area of the image into gray,
then send back the editted image to a client.
"""
from bottle import route, post, run, template, request, hook, static_file, HTTPResponse
import io
import cv2
import numpy as np
import os
import sys
import time
import base64
import image_processor
sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn'))
from mask_rcnn.mrcnn_model import MaskRcnnModel

# サーバー起動時にモデルインスタンス作成
mrcnn_model = MaskRcnnModel()

@route('/health')
def health():
    return 

@post('/splash/gray')
def gray():
    ###
    #keep human area color and make other area gray
    ###
    s = time.time()
    upload = request.files.get('file_1') #html inputフィールドのname属性を指定
    f = io.BytesIO()
    upload.save(f)
    img = cv2.imdecode(np.frombuffer(f.getbuffer(), np.uint8), 1) #image with color(3 channels)
    # cv2.imwrite('debug.jpg', img)
    decoding_fin = time.time()
    img_decoding_time = decoding_fin - s
    print('decoding time: ', img_decoding_time)
    result = mrcnn_model.detect(img)
    res_img = image_processor.make_image_gray(result, img)
    # res_img = mrcnn_model.detect(img)
    name = os.path.splitext(upload.filename)[0]
    cv2.imwrite('{}_gray.jpg'.format(name), res_img)

    detect_fin = time.time()
    detecting_time = detect_fin - decoding_fin
    print('detecting time: ', detecting_time)

    _, res_str = cv2.imencode('.jpg', res_img)
    res_str = base64.b64encode(res_str)

    res = HTTPResponse(status=200, body=res_str)
    res.set_header('Content-Type', 'image/jpg')
    res.set_header('Access-Control-Allow-Origin', '*') # cross domain

    return res

@post('/splash/blur')
def blur():
    ###
    #keep human area focused and make other area blur
    ###
    s = time.time()
    upload = request.files.get('file_1') #html inputフィールドのname属性を指定
    f = io.BytesIO()
    upload.save(f)
    img = cv2.imdecode(np.frombuffer(f.getbuffer(), np.uint8), 1) #image with color(3 channels)
    decoding_fin = time.time()
    img_decoding_time = decoding_fin - s
    print('decoding time: ', img_decoding_time)
    result = mrcnn_model.detect(img)
    res_img = image_processor.make_image_blur(result, img)

    detect_fin = time.time()
    detecting_time = detect_fin - decoding_fin
    print('detecting time: ', detecting_time)

    _, res_str = cv2.imencode('.jpg', res_img)
    res_str = base64.b64encode(res_str)

    res = HTTPResponse(status=200, body=res_str)
    res.set_header('Content-Type', 'image/jpg')
    res.set_header('Access-Control-Allow-Origin', '*') # cross domain

    return res

@post('/splash/blur_gray')
def blur_gray():
    ###
    #keep human area focused and make other area blur
    ###
    s = time.time()
    upload = request.files.get('file_1') #html inputフィールドのname属性を指定
    f = io.BytesIO()
    upload.save(f)
    img = cv2.imdecode(np.frombuffer(f.getbuffer(), np.uint8), 1) #image with color(3 channels)
    decoding_fin = time.time()
    img_decoding_time = decoding_fin - s
    print('decoding time: ', img_decoding_time)
    result = mrcnn_model.detect(img)
    res_img = image_processor.make_image_blur_gray(result, img)

    detect_fin = time.time()
    detecting_time = detect_fin - decoding_fin
    print('detecting time: ', detecting_time)

    _, res_str = cv2.imencode('.jpg', res_img)
    res_str = base64.b64encode(res_str)

    res = HTTPResponse(status=200, body=res_str)
    res.set_header('Content-Type', 'image/jpg')
    res.set_header('Access-Control-Allow-Origin', '*') # cross domain

    return res

#trial of downloader
@route('/images/<file_path:path>')
def static(file_path):
    return static_file(file_path, root='./', download=True)


run(host='0.0.0.0', port=8080)