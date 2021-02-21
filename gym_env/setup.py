from distutils.core import setup

setup(name='blacky_bullet',
      version='1.0',
      description='A package to run reinforcement learning algorithms on pybullet',
      author='Eduardo López and Amin Arriaga',
      author_email='eduardo98m@gmail.com',
      url='',
      packages=['gym_env', 'ars_agent', 'bullet_dataclasses'],
     )