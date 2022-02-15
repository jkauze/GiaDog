@echo off
docker run  --rm -it -v "%cd%"/src:/usr/src/GiaDog/src/spot_mini_ros giadog
