# Design and Benchmarking of an Onboard AI for Multi-Wavelength Solar Structure Segmentation

This work addresses these limitations by developing lightweight deep learning models for the segmentation of active regions, coronal holes, and filaments in extreme ultraviolet images. By extending existing datasets with our filament dataset, we propose a multi-wavelength, multi-label dataset. Using a self-adaptable U-Net-based segmentation scheme, various settings and strategies were evaluated, resulting in ultralight multi-class multi-wavelength U-Net models

![Example of active regions and coronal holes segmentation by the networks the 3-stage with Dice+CE loss on the datasets 199 and 364.](docs/199_4_display_pred_case_0080.pdf)
![](docs/364_4_display_pred_case_0020.pdf)
![](docs/671_4_display_pred_case_0010.pdf)


## Installation of the nnUNet framework

Follow the [nnUNet](https://github.com/mic-dkfz/nnunet) github for installation.


Then, 
```bash
git clone https://github.com/Erthryx/Onboard-solar-structure-segmentation-network/

```
The following folders contain:

- **code**: This folder contains all the code developed and used throughout this work, including preprocessing, training, inference, and evaluation scripts.

- **data**: This folder contains all datasets used in this work. Their structure follows the nnU-Net format, allowing them to be directly integrated into an nnU-Net environment. Each dataset is accompanied by a dedicated README file describing its parameters, the channels composing the multi-wavelength input, and the labels included in the final prediction mask.

- **models**: This folder contains all trained neural network models developed and evaluated in this work.



Original datasets that were extended:
[SCSS-net](https://github.com/space-lab-sk/scss-net)
[DETACH](https://github.com/Junyan-L/DETACH)
