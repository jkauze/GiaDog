FROM ros:melodic

SHELL ["/bin/bash", "-c"]

# Create work directory
ENV OPEN_BLACKY_DIR '/usr/src/open_blacky'
RUN mkdir -p ${OPEN_BLACKY_DIR}
WORKDIR ${OPEN_BLACKY_DIR}

# Install dependencies
RUN apt update && \
    apt install -y python3-pip python3-catkin-pkg-modules python3-rospkg-modules && \
    python3 -m pip install --upgrade setuptools pip wheel && \
    python3 -m pip install dataclasses gym matplotlib numpy perlin-noise pybullet 
