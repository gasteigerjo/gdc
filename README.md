# GDC

<p align="center">
<img src="https://raw.githubusercontent.com/gasteigerjo/gdc/master/fig_model.svg?sanitize=true" width="600">
</p>

Reference implementation (example) of the model proposed in the paper:

**[Diffusion Improves Graph Learning](https://www.kdd.in.tum.de/gdc)**   
by Johannes Gasteiger, Stefan Weißenberger, Stephan Günnemann   
Published at NeurIPS 2019.

## Run the code
This repository primarily contains a demonstration of enhancing a graph convolutional network (GCN) with graph diffusion convolution (GDC) in the notebook `gdc_demo.ipynb`.

## Requirements
The repository uses these packages:

```
pyyaml
tqdm>=4.36
numpy
scipy
seaborn
pytorch>=1.3
pytorch_geometric
```

## PyTorch Geometric

GDC is also implemented as a transformation (preprocessing step) in [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GDC). So you can just apply it to your own dataset and see how your existing PyG model improves!

## Contact
Please contact j.gasteiger@in.tum.de in case you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{gasteiger_diffusion_2019,
  title = {Diffusion Improves Graph Learning},
  author = {Gasteiger, Johannes and Wei{\ss}enberger, Stefan and G{\"u}nnemann, Stephan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2019}
}
```
