FROM ros:melodic

# Install dependencies
RUN apt update && \
    apt install -y python3-pip && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy pybullet gym matplotlib

RUN mkdir -p /usr/src/open-blacky
WORKDIR /usr/src/open-blacky