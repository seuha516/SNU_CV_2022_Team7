FROM tensorflow/tensorflow
WORKDIR /app
COPY . /app

RUN apt-get update
RUN apt-get install build-essential cmake libgl1-mesa-glx libglib2.0-0 -y
RUN python -m transformers.onnx --model=bert-base-cased onnx/bert-base-cased/
RUN mv u2net.onnx /root/.u2net