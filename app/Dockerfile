FROM tensorflow/tensorflow:1.13.1-py3
# 必要なファイルをappディレクトリへコピー
COPY ./ /app

# appディレクトリをルート設定
WORKDIR /app

RUN apt update &&\
    apt install -y uwsgi-plugin-python3 wget libsm6 libxext6 libxrender-dev libglib2.0-0

RUN pip3 install -r requirements.txt

RUN sh prep.sh

# uwsgiに必要なファイル設定
RUN mkdir /var/log/uwsgi && \
    chown www-data:www-data /var/log/uwsgi

# CMD ["uwsgi", "--ini", "uwsgi.ini"]
CMD ["bash"]