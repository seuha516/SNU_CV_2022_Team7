FROM tensorflow/tensorflow:latest
WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install build-essential cmake libgl1-mesa-glx libglib2.0-0 -y
RUN pip install -r requirements.txt
RUN python -m transformers.onnx --model=bert-base-cased onnx/bert-base-cased/
RUN mkdir /root/.u2net/
RUN mv u2net.onnx /root/.u2net/