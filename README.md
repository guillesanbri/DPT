# Vision Transformers for Dense Prediction

This repository contains code and models for the Master's Thesis "Estimación de Profundidad Online con Transformers Eficientes", which modifies the code published along the [paper](https://arxiv.org/abs/2103.13413) presented by Ranftl et al. to accelerate the inference speed of the Dense Prediction Transformers.

## Related links
- [Models and results from the project](https://zenodo.org/record/6574941)
- [Manuscript repository](https://github.com/guillesanbri/tfm-latex/tree/v1.0.0)

## Abstract

Monocular depth estimation deals with the automatic recovery of an approximation of the dimension that is lost when projecting a three-dimensional scene into a two-dimensional image. This problem has an infinite number of geometric solutions, which makes it practically impossible to solve using traditional computer vision techniques. Nonetheless, Deep Learning techniques are capable of extracting different characteristics from the images that make it possible to approximate a solution. In this work this problem and the existing solutions are studied, especially those based on Transformers and supervised learning. In one of these solutions, a series of modifications and developments are carried out to reduce the size of the original model and multiply its inference speed by nearly five. Furthermore, an exhaustive study, both quantitative and qualitative, of the influence of the different modifications is included, evaluating the models in the KITTI dataset, oriented to autonomous driving.

## Documentation

Documentation for this project can be found in the Appendix B of the [Master's Thesis manuscript](https://github.com/guillesanbri/tfm-latex/blob/v1.0.0/main.pdf) (ES).

### Acknowledgements

This work obviously would not have been possible without the incredibly valuable contribution of the [Vision Transformers for Dense Prediction](https://github.com/isl-org/DPT) paper and the implementations of efficient attention mechanisms from [Phil Wang](https://github.com/lucidrains). Likewise, a huge thank you to the PyTorch community and Ross Wightman for his incredible work with [timm](https://github.com/rwightman/pytorch-image-models).

### License

MIT License 
