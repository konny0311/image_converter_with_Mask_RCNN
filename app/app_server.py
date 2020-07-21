# -*- coding: utf-8 -*-
"""
This is an API server for "splash of color" using Mask R-CNN.
The server recieves an image and turns non-human area of the image into gray,
then send back the editted image to a client.
"""
import io
import os
import cv2
import sys
import time
import json
import base64
import secrets
import datetime
import requests
import numpy as np
import image_processor
sys.path.append(os.path.join(os.getcwd(), 'mask_rcnn'))
from mask_rcnn.mrcnn_model import MaskRcnnModel
from google.cloud import storage
from bottle import route, post, run, template, request, hook, static_file, HTTPResponse

# サーバー起動時にモデルインスタンス作成
mrcnn_model = MaskRcnnModel()
storage_client = storage.Client()
BUCKET = storage_client.bucket('line-konny')

@route('/')
def health():
    return 

@post('/splash/line')
def line():
    """
    json = {'image_url': image_url
            'reply_token': request['replyToken']}
    ステータスコード200とコンテンツのバイナリデータを返します。
    how to get an image, see more https://developers.line.biz/ja/reference/messaging-api/#get-content
    """
    image_url = request.forms.get('image_url')
    print('image_url', image_url)
    reply_token = request.forms.get('reply_token')
    print('reply_token', reply_token)
    convert_type = request.forms.get('convert_type')
    headers = {'Authorization': 'Bearer {}'.format(ACCESS_TOKEN)}
    res = requests.get(image_url, headers=headers)
    #res.contentでバイナリデータ受け取る
    f = io.BytesIO(res.content)
    img = cv2.imdecode(np.frombuffer(f.getbuffer(), np.uint8), 1) #image with color(3 channels)
    result = mrcnn_model.detect(img)
    if convert_type == 'blur':
        img = image_processor.make_image_blur(result, img)
    elif convert_type == 'gray':
        img = image_processor.make_image_gray(result, img)
    elif convert_type == 'blur_gray':
        img = image_processor.make_image_blur_gray(result, img)
    now = datetime.datetime.now()
    name = now.strftime('%Y%m%d%H%M%S')
    ran = secrets.token_hex(16)
    filename = '{}_blur_{}.jpg'.format(name, ran)
    print('{} saved'.format(filename))

    _save_img2cloudStorage(img, filename)

    thumbnail_img = _create_thumbnail(img)
    thumbnail_filename = '{}_thumbnail_blur_{}.jpg'.format(name, ran)
    _save_img2cloudStorage(thumbnail_img, thumbnail_filename)
                                     
    image_url_after_convert = 'https://storage.googleapis.com/{}/{}'.format(BUCKET, filename)
    thumbnail_image_url_after_convert = 'https://storage.googleapis.com/{}/{}'.format(BUCKET, thumbnail_filename)
    message = {
                "type": "image",
                "originalContentUrl": image_url_after_convert,
                "previewImageUrl": thumbnail_image_url_after_convert
              }

    line_url = 'https://api.line.me/v2/bot/message/reply'
    headers = {'Content-Type':'application/json',
               'Authorization': 'Bearer {}'.format(ACCESS_TOKEN)}
    body = {'replyToken':reply_token,
            'messages': [message]}
    body_json = json.dumps(body)
    print(body_json)
    line_res = requests.post(line_url, data=body_json, headers=headers)
    print(line_res)
    print(line_res.json())

    res = HTTPResponse(status=200)

    return res


#trial of downloader
@route('/images/<file_path:path>')
def static(file_path):
    return static_file(file_path, root='./', download=True)


def _create_thumbnail(img):
    """
    line api accepts 240 x 240 thumbnail image as max
    """
    h, w, _ = img.shape
    if h > w:
        w = int(w * (240/h))
        h = 240
    else:
        h = int(h * (240/w))
        w = 240
    
    return cv2.resize(img, (w, h))

def _save_img2cloudStorage(img, filename):
    _, img_str = cv2.imencode('.jpg', img)
    img_bytes = img_str.tobytes()
    blob = BUCKET.blob(filename)
    blob.upload_from_string(img_bytes)
    blob.make_public()

run(host='0.0.0.0', port=8080)
