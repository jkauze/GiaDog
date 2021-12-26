<div id="top"></div>
<!--
*** REFERENCES: https://github.com/othneildrew/Best-README-Template
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/eduardo98m/open-blacky">
    <img src="images/logo.png" alt="Logo [TODO]" width="80" height="80">
  </a>

  <h3 align="center">Open Blacky</h3>

  <p align="center">
    Simulation environment for training spot-mini robots in quadruped locomotion using reinforced learning
    <br />
    <a href="https://github.com/eduardo98m/open-blacky"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/eduardo98m/open-blacky">View Demo</a>
    ·
    <a href="https://github.com/eduardo98m/open-blacky/issues">Report Bug</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<summary>Table of Contents</summary>
<ol>
<li>
    <a href="#about-the-project">About The Project</a>
    <ul>
    <li><a href="#built-with">Built With</a></li>
    </ul>
</li>
<li>
    <a href="#getting-started">Getting Started</a>
    <ul>
    <li><a href="#prerequisites">Prerequisites</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#google-colab">Google Colab</a></li>
    </ul>
</li>
<li><a href="#usage">Usage</a></li>
<li><a href="#license">License</a></li>
<li><a href="#contact">Contact</a></li>
</ol>

<!-- ABOUT THE PROJECT -->
## About The Project

[TODO]

### Built With

* [Python](https://www.python.org/)
* [Docker](https://www.docker.com/)
* [ROS Melodic Morenia](http://wiki.ros.org/melodic)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* Docker >=20.10.12

### Installation

1. Build the docker image.

```bash
docker build . -t open-blacky
```

2. Run the container of said image. It is necessary to allow the container to connect
 with the host's display in order to view the simulation

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
    -v $PWD/src:/usr/src/open_blacky/src/spot_mini_ros \
    open-blacky
```

3. Once inside the container, build the ROS package

```bash
source /opt/ros/melodic/setup.bash 
catkin_make_isolated
echo -e "\nsource /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo -e "source ${OPEN_BLACKY_DIR}/devel_isolated/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Google Colab

The repository can also be run in [Google Colab](https://colab.research.google.com/) 
following the instructions in the following 
[notebook](https://colab.research.google.com/drive/1I88SeRK-xUmy_r_ZAUL5AZcPFmv57xnI?usp=sharing). 
With the limitation that the simulation GUI cannot be executed since Google Colab does 
not have X server and the sessions (free) are limited to 9 continuous hours maximum.


<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

```bash
roscore > /dev/null &
rosrun spot_mini_ros simulate.py <ARGS>
```

### Terrain generation

The generation of terrain in a simulation allows the agent to learn to move in various
situations before moving on to the real robot. The terrains are stored as `.txt` files 
that contain an matrix such that each position indicates the height of the terrain in 
that pixel. There are 3 types of terrain:

#### Hills

They represent a rugged terrain with mountains, very similar to a rocky environment, 
which will allow the agent to learn to move in unstable places. These terrains are 
defined with three parameters: roughness, frequency and amplitude.

`Easy (roughness=0.0, frequency=0.2, amplitude=0.2)`

![Hills easy](docs/terrain_examples/hills_easy.png) 

`Medium (roughness=0.02, frequency=1.6, amplitude=1.6)`

![Hills medium](docs/terrain_examples/hills_medium.png) 

`Hard (roughness=0.04, frequency=3.0, amplitude=3.0)`

![Hills hard](docs/terrain_examples/hills_hard.png) 

#### Steps 

It represents a terrain of uneven cubes, being able to represent an environment where 
there are many obstacles on the ground. These terrains are defined with three parameters: 
width and max height of cubes.

`Easy (width=25, height=0.05)`

![Steps easy](docs/terrain_examples/steps_easy.png)

`Medium (width=17, height=0.23)`

![Steps medium](docs/terrain_examples/steps_medium.png)

`Hard (width=10, height=0.4)`

![Steps hard](docs/terrain_examples/steps_hard.png)

#### Stairs

It places the agent in the middle of some stairs, forcing him to learn to go up and down 
them to achieve various objectives. These terrains are defined with three parameters: 
width and height of steps.

`Easy (width=50, height=0.02)`

![Stairs easy](docs/terrain_examples/stairs_easy.png) 

`Medium (width=30, height=0.11)`

![Stairs medium](docs/terrain_examples/stairs_medium.png)

`Hard (width=15, height=0.2)`

![Stairs hard](docs/terrain_examples/stairs_hard.png)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

* Amin Arriaga - aminlorenzo.14@gmail.com
* Eduardo López - eduardo98m@gmail.com

Project Link: [https://github.com/eduardo98m/open-blacky](https://github.com/eduardo98m/open-blacky)

<p align="right">(<a href="#top">back to top</a>)</p>

https://github.com/eduardo98m/open-blacky
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/eduardo98m/open-blacky.svg?style=for-the-badge
[contributors-url]: https://github.com/eduardo98m/open-blacky/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/eduardo98m/open-blacky.svg?style=for-the-badge
[forks-url]: https://github.com/eduardo98m/open-blackye/network/members
[stars-shield]: https://img.shields.io/github/stars/eduardo98m/open-blacky.svg?style=for-the-badge
[stars-url]: https://github.com/eduardo98m/open-blacky/stargazers
[issues-shield]: https://img.shields.io/github/issues/eduardo98m/open-blacky.svg?style=for-the-badge
[issues-url]: https://github.com/eduardo98m/open-blackye/issues
[license-shield]: https://img.shields.io/github/license/eduardo98m/open-blacky.svg?style=for-the-badge
[license-url]: https://github.com/eduardo98m/open-blackye/blob/master/LICENSE.txt
