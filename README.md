[한국어](./README_kr.md)


# practice-FirstPersonOpenGL

![Overview](/screenshots/01.png)

Simple Python-OpenGL project to study how to use it.
There are nothing fancy things to do here, just walking/flying around, looking around, colliding to aabb boxes.

There are 3 kinds of basics light sources (point, spot, directional) and for directional light it can cast shadow.
On the center of the room, you can see the shadow map used by the directional light.

It can import 3D models in OBJ format. Press F8 key to load huge city map.
Since I didn't implemented multithreading, the app freezes while loading.
Z-fighting isn't addressed correctly so the blue plane might look bad.
Also, loading the huge model makes shadow-casting area larger, which makes shadow look more blocky.


# How to run the program

Clone the repository and run `{git_repo}/main_v2.py`.
Current working directory must be `{git_repo}` or it fails to load shader source files.


# Dev Environment

* Python 3.5
* Windows 10 (x64)

#### Needed Libraries

* Numpy
* PyGame
* PyOpenGL
* Pillow


# Control

* WASD : Horizontal movement
* ↑←↓→ : Look around
* SPACE : Vertical ascent (on noclip mode)
* SHIFT : Vertical Descent (on noclip mode)
* F : Toggle flashlight
* F6 : Toggle look around with mouse (default is off)
* F7 : Toggle noclip (default is off)
* F8 : Load city map model (it freezes the app for a few seconds)
* F10 : Increase orbit speed of boxes by 10 (default is 20)
* F11 : Decrease orbit speed of boxes by 10
* ESC : Exit app
