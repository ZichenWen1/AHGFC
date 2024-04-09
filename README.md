# Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph Clustering
Zichen Wen, Yawen Ling, Yazhou Ren, Tianyi Wu, Jianpeng Chen, Xiaorong Pu, Zhifeng Hao, Lifang He

This is the source code for the paper: Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph Clustering, accepted by AAAI 2024.

 </div>
<div align="center">
    <a href="https://arxiv.org/pdf/2401.02682.pdf"><img src="Paper-Arxiv-orange.svg" ></a>
</div>


## Requirements

- requirements.txt
- Environment
  - NVIDIA GeForce GTX 1080Ti (GPU)
  - CUDA version: 11.6
  - torch version: 1.13.1 
  - 12th Gen Intel(R) Core(TM) i7-12700KF (CPU)



## Datasets

|  Dataset  | #Clusters | #Nodes | #Features |                           Graphs                            |              HR              |
| :-------: | :-------: | :----: | :-------: | :---------------------------------------------------------: | :--------------------------: |
|   Texas   |     5     |  183   |   1703    |            $\mathcal{G}^1$ <br />$\mathcal{G}^2$            |       0.09 <br />0.09        |
| Chameleon |     5     |  2777  |   2325    |            $\mathcal{G}^1$ <br />$\mathcal{G}^2$            |       0.23 <br />0.23        |
|    ACM    |     3     |  3025  |   1830    |            $\mathcal{G}^1$ <br />$\mathcal{G}^2$            |       0.82 <br />0.64        |
| Wiki-cooc |     5     |  10000 |   100     |                       $\mathcal{G}^1$                       |             0.34             |
|Minesweeper|     2     |  10000 |    7      |                       $\mathcal{G}^1$                       |             0.68             |
|  Workers  |     2     |  11758 |    10     |                       $\mathcal{G}^1$                       |             0.59             |

## Test AHGFC

```python
# Test AHGFC
python AHGFC_test.py
```

## Train AHGFC

```python
# Train AHGFC
python AHGFC_train.py

```

## Citation
If you're using AHGFC in your research or applications, please cite using this BibTeX:

```bibtex
@inproceedings{wen2024homophily,
  author       = {Zichen Wen and
                  Yawen Ling and
                  Yazhou Ren and
                  Tianyi Wu and
                  Jianpeng Chen and
                  Xiaorong Pu and
                  Zhifeng Hao and
                  Lifang He},
  title        = {Homophily-Related: Adaptive Hybrid Graph Filter for Multi-View Graph
                  Clustering},
  booktitle    = {AAAI},
  pages        = {15841--15849},
  year         = {2024}
}
```


