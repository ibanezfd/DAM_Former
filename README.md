# Inter-Sensor High-Resolution and Multi-Temporal Image Fusion for Unsupervised Domain Adaptation in Remote Sensing

[Damián Ibáñez Fernández](https://orcid.org/0000-0002-3252-1252), [Junshi Xia](https://scholar.google.com/citations?user=n1aKdTkAAAAJ&hl=en), [Naoto Yokoya](https://naotoyokoya.com/), [Rubén Fernández Beltrán](https://scholar.google.es/citations?user=pdzJmcQAAAAJ&hl=es), [Filiberto Pla Bañón](https://ieeexplore.ieee.org/author/37270640500)

___________

This repository includes code related to the article ["Inter-Sensor High-Resolution and Multi-Temporal Image Fusion for Unsupervised Domain Adaptation in Remote Sensing"](https://ieeexplore.ieee.org/document/11000318). 

Motivated by the increasing demand for robust segmentation in unlabeled remote sensing data, we propose domain adaptation multimodal and multi-temporal transformer (DAM-Former), a novel unsupervised domain adaptation (UDA) model that fuses high-resolution (HR) multimodal imagery with multi-temporal multispectral data. Current UDA approaches in remote sensing rarely exploit the complementary strengths of spatial and temporal features. To address this gap, our framework integrates two interconnected branches: a transformer-based network for HR multimodal data and a lightweight convolutional network with temporal attention for multi-temporal imagery. To improve segmentation accuracy and lower noise, the extracted features are robustly combined through a deep temporal fusion (DTF) module and a new mixed loss (ML) with an ensemble pseudo-label (EP) strategy. Extensive experiments and an ablation study on the FLAIR-2 dataset demonstrate that DAM-Former outperforms state-of-the-art methods, marking the first in-depth study of temporal information fusion in UDA segmentation for remote sensing data.

![Architecture of the DAM-Former. On the top, the low-resolution (LR) multitemporal branch, with the convolutional encoder and decoder blocks in blue and pink, respectively. Including the temporal attention block in green and the deep temporal fusion module in purple. On the bottom, the high-resolution (HR) multimodal branch, with the transformer encoders in yellow, and the lighter convolutional decoders in red.](images/DAMFORMER.png)

![South to North UDA segmentation visual results covering an urban area with a large number of buildings, roads, parks and a river, where: (a) Ground Truth, (b) RGB image, (c) NIR image, (d) elevation image, (e) RGB U-Net (C), (f) RGB DAFormer, (g) RGB U-Net (T), (h) RGB+NIR+E U-Net (C), (i) RGB+NIR+E DAFormer, (j) RGB+NIR+E U-Net (T), (k) RGB+T U-Net (C), (l) RGB+T DAFormer, (m) RGB+T U-Net (T), (n) RGB+NIR+E+T U-Net (C), (o) RGB+NIR+E+T DAFormer, (p) RGB+NIR+E+T \textbf{DAM-Former}.](images/example.png)

Citation
---------------------

**Please kindly cite the paper if this code is useful and helpful for your research.**

Ibañez, D., Xia, J., Yokoya, N., Pla, F., & Fernandez-Beltran, R. (2025). Inter-sensor High-Resolution and Multi-temporal Image Fusion for Unsupervised Domain Adaptation in Remote Sensing. IEEE Transactions on Geoscience and Remote Sensing.

    @article{ibanez2025inter,
    title={Inter-sensor High-Resolution and Multi-temporal Image Fusion for Unsupervised Domain Adaptation in Remote Sensing},
    author={Iba{\~n}ez, Damian and Xia, Junshi and Yokoya, Naoto and Pla, Filiberto and Fernandez-Beltran, Ruben},
    journal={IEEE Transactions on Geoscience and Remote Sensing},
    year={2025},
    publisher={IEEE}
    }

System-specific notes
---------------------
The codes and networks were tested using PyTorch 1.7.1 version (CUDA 11.0) in Python 3.8.5 on the supercomputer RAIDEN provided by [RIKEN AIP](https://www.riken.jp/) using an Nvidia Tesla V100 and a CPU Intel1 Xeon1 E5-2698 v4.

Data and code notes
---------------------
The data used in this paper can be found in [FLAIR #2](https://github.com/IGNF/FLAIR-2). 

The code for this work was based on the code in [DAFormer](https://github.com/lhoyer/DAFormer). In this repository we included our model, DAM-Former and the necessary modified files to use our model within their framework.

Dataset preparation
---------------------
In order to use the FLAIR #2 dataset, first run the script [flair2.py]() to generate the required headers. In this file it is possible to select different domains of the FLAIR #2 dataset. You should run it for the source domain(s) and target domain(s) to be used. 
Then, follow the steps described in [DAFormer](https://github.com/lhoyer/DAFormer) with our modifications and model. 

Licensing
---------

Copyright (C) 2025 Damián Ibáñez

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

Contact Information:
--------------------

Damián Ibáñez: ibanezd@uji.es<br>
Damián Ibáñez is with the Institute of New Imaging Technologies, Universitat Jaume I, 12071 Castellón de la Plana, Spain. 
