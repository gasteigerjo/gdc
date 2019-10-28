# GDC

<p align="center">
<img src="https://raw.githubusercontent.com/klicperajo/gdc/master/fig_model.svg?sanitize=true" width="600">
</p>

Reference implementation (example) of the model proposed in the paper:

**[Diffusion Improves Graph Learning](https://www.kdd.in.tum.de/gdc)**   
by Johannes Klicpera, Stefane Weißenberger, Stephan Günnemann   
Published at NeurIPS 2019.

## Run the code
This repository primarily contains the example of using graph diffusion convolution (GDC) with GCN in the notebook `gdc_demo.ipynb`.

## Requirements
The repository uses these packages:

```
numpy
scipy
pytorch>=1.0
pytorch_geometric
tqdm
seaborn
```

## Contact
Please contact klicpera@in.tum.de in case you have any questions.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@inproceedings{klicpera_diffusion_2019,
  title = {Diffusion Improves Graph Learning},
  author = {Klicpera, Johannes and Wei{\ss}enberger, Stefan and G{\"u}nnemann, Stephan},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year = {2019}
}
```
