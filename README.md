# 3DFace_DaViT_Blender - CoFiT-3D FaRe

**📌 Please note: This project is no longer actively maintained. It remains public as a reference for those interested, but dependencies or installation steps may be outdated.**

this repo are available for non-commercial scientific research purposes only.

3D face reconstruction from single image using Dual Vision Transformer network and plugin for Blender

Paper: [A New Coarse-To-Fine 3D Face Reconstruction Method Based On 3DMM Flame and Transformer: CoFiT-3D FaRe](https://dl.acm.org/doi/10.1145/3628797.3628960)

<p align="center"> 
<img src="images/texture_result.png" width="500">
</p>
<p align="center">Our texture reconstruction result. From left to right of each view: direct projected texture, fine texture, coarse 3DMM texture.<p align="center">

<p align="center"> 
<img src="images/shape_result.png" width="800">
</p>
<p align="center">Our shape reconstruction results.<p align="center">

Demo Video: https://www.youtube.com/watch?v=qp1f-0poktE

## Introduction

This source code enables the reconstruction of a three-dimensional human face model from a two-dimensional input image. The model employs the 3DMM Flame and the deep learning network DaViT. Moreover, the source code includes a plugin within the Blender software, enhancing its usability for users.


## Acknowledgements

We would like to express our sincere gratitude to the following projects and their contributors:

- **DECA Repository**: The development of this project has been inspired by and built upon the DECA repository, which provided valuable insights and resources that accelerated our work. The DECA repository can be found [here](https://github.com/yfeng95/DECA).

- **3DMM FLAME Model**: The Three-Dimensional Morphable Model (3DMM) FLAME has been instrumental in our research and development efforts. We acknowledge the creators of the 3DMM FLAME model for their pioneering work. More information about the 3DMM FLAME model can be accessed [here](https://flame.is.tue.mpg.de/).

-  **DaViT**: The Dual Vision Transformer network (DaViT) serves as the backbone network of our deep learning model and has significantly contributed to the capabilities of our project. We express our appreciation to the developers of DaViT for their invaluable contributions. For more details about DaViT, visit [here](https://github.com/dingmyu/davit).


We also thank all the open-source contributors and the broader community for their continuous support and contributions to the field of computer graphics and computer vision. Without the hard work and dedication of the aforementioned projects and their contributors, our project would not have been possible. We are indebted to their groundbreaking work.


## Citation 

Please cite our github repo if you find our work helpful.

