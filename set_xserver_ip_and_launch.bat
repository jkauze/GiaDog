@echo off
Title set docker container xserver ip
for /f "tokens=3 delims=: " %%i  in ('netsh interface ip show config name^="vEthernet (WSL)" ^| findstr "IP Address" ^| findstr [0-9]') do set MY_IP=%%i
echo Network IP : %MY_IP%
for /f %%i in ('docker ps --format {{.Names}}') do set containerName=%%i
echo Container name :  %containerName%
docker exec %containerName% bash -c "echo 'export DISPLAY=%MY_IP%:0.0' >> ~/.bashrc"
docker exec %containerName% bash -c "source ~/.bashrc"
echo Vxsver IP adress setted to the container now GUI applications are accessible
docker exec -it %containerName% bash 