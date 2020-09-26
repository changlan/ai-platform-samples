FROM gcr.io/deeplearning-platform-release/tf-gpu.1-15

# Install OpenSSH, CUDA 10.0 and NCCL
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends --allow-change-held-packages \
        build-essential cmake iperf openssh-client openssh-server cuda-toolkit-10-0 libnccl2 libnccl-dev libopenmpi-dev openmpi-bin && \
    rm -rf /var/lib/apt/lists/* && \
    ldconfig /usr/local/cuda/targets/x86_64-linux/lib

# Install Horovod.
# pyarrow is broken in the DL container base image (b/150726325)
RUN HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 HOROVOD_WITHOUT_GLOO=1 \
         pip install --no-cache-dir horovod[tensorflow] && \
         pip install --no-cache-dir --force-reinstall pyarrow && \
    ldconfig

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Create SSH key.
RUN mkdir -p /root/.ssh/ && \
    mkdir -p /var/run/sshd && \
    ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
    cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys

WORKDIR /infra

# Use along with
# ENV GCS_OUTPUT_PATH gs://path
# to copy gs://path to /output
ENV OUTPUT_DIR /output/
EXPOSE 2222

COPY horovod_wrapper.py /infra

# Use along with ENV STAGE_GCS_PATH gs://path to copy gs://path to /output
ENV STAGING_DIR /input

# For local run only
ENV TF_CONFIG '{"cluster":{"master":["127.0.0.1:2222"]}, "task":{"type":"master","index":0}}'

ENV NCCL_DEBUG INFO
ENV NCCL_FAST_SOCKET_ENABLE 0

ENTRYPOINT ["python", "/infra/horovod_wrapper.py"]
