# **Graduation Thesis: GIAdog**

## **Pre-requisitos**

 * `g++ >=8.3.0`
 * `python >=3.7.0`

## **Ejecutar Docker**

```bash
xhost +
sudo docker run \
    --device /dev/dri/ \
    --device /dev/snd \
    --env="QT_X11_NO_MITSHM=1" \
    --ipc=host \
    --net=host \
    --rm \
    -e _JAVA_AWT_WM_NONREPARENTING=1 \
    -e DISPLAY=$DISPLAY \
    -e J2D_D3D=false \
    -it \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $PWD:/usr/src/open-blacky \
    open-blacky
```

## **Generacion de terrenos**

La generacion de terrenos en una simulacion permite al agente GIAdog aprender a moverse
antes de pasar al robot real. Los terrenos son almacenados como archivos `.txt` que 
contienen una matriz tal que cada posicion indica la altura del terreno en dicho pixel.

### **Hills**

Easy: 0 0.2 0.2

![Hills easy](docs/terrain_examples/hills_easy.png) 

Medium: 0.02 1.6 1.6

![Hills medium](docs/terrain_examples/hills_medium.png) 

Hard: 0.04 3 3

![Hills hard](docs/terrain_examples/hills_hard.png) 

### **Steps** 

Easy: 25, 0.05

![Steps easy](docs/terrain_examples/steps_easy.png)

Medium 17, 0.23

![Steps medium](docs/terrain_examples/steps_medium.png)

Hard: 10, 0.4

![Steps hard](docs/terrain_examples/steps_hard.png)

### **Stairs**

Easy: 50, 0.02

![Stairs easy](docs/terrain_examples/stairs_easy.png) 

Medium: 30, 0.11

![Stairs medium](docs/terrain_examples/stairs_medium.png)

Hard: 15, 0.2

![Stairs hard](docs/terrain_examples/stairs_hard.png)

## **AUTORES**

 * Amin Arriaga
 * Eduardo Lopez