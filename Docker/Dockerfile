FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ARG USER_ID
ARG GROUP_ID

WORKDIR /workspace

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN python3 -m pip install --upgrade pip

COPY DPT/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

ENTRYPOINT ["/bin/bash"]
