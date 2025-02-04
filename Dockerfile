FROM ros:noetic

SHELL ["/bin/bash", "-c"]

# Create work directory
ENV GIADOG_DIR '/usr/src/GiaDog'
RUN mkdir -p ${GIADOG_DIR}
WORKDIR ${GIADOG_DIR}
COPY .env.json ${GIADOG_DIR}/

# Install dependencies
RUN (apt update && \
        apt install -y python3-pip python3-catkin-pkg-modules python3-rospkg-modules && \
        python3 -m pip install --upgrade setuptools pip wheel && \
        python3 -m pip install dataclasses gym matplotlib numpy perlin-noise pybullet \
            tensorflow keras-tcn nvidia-cuda-toolkit) || \
    while ! [[ $? -eq "0" ]]; do \
        apt update && \
        apt install -y python3-pip python3-catkin-pkg-modules python3-rospkg-modules && \
        python3 -m pip install --upgrade setuptools pip wheel && \
        python3 -m pip install dataclasses gym matplotlib numpy perlin-noise pybullet \
            tensorflow keras-tcn nvidia-cuda-toolkit; \
    done

COPY .env.json ${GIADOG_DIR}/
