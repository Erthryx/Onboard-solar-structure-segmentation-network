# Design and Benchmarking of an Onboard AI for Multi-Wavelength Solar Structure Segmentation

This work addresses these limitations by developing lightweight deep learning models for the segmentation of active regions, coronal holes, and filaments in extreme ultraviolet images. By extending existing datasets with our filament dataset, we propose a multi-wavelength, multi-label dataset. Using a self-adaptable U-Net-based segmentation scheme, various settings and strategies were evaluated, resulting in ultralight multi-class multi-wavelength U-Net models

![Example of active regions and coronal holes segmentation by the networks the 3-stage with Dice+CE loss on the datasets 199 and 364.](docs/demo.gif)


## Installation of the nnUNet framework

Follow the [nnUNet](https://github.com/mic-dkfz/nnunet) github for installation.


Then, 
```bash
git clone https://github.com/Erthryx/Onboard-solar-structure-segmentation-network/
´´´

The 

Original datasets that were extended:
[SCSS-net](https://github.com/space-lab-sk/scss-net)
[DETACH](https://github.com/Junyan-L/DETACH)
