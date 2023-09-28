# Manifold Path Guiding

Zhimin Fan*, Pengpei Hong*, Jie Guo, Changqing Zou, Yanwen Guo, and Ling-Qi Yan

_ACM Transactions on Graphics (Proceedings of SIGGRAPH ASIA 2023)_

![ManifoldPG_teaser_720p](img/ManifoldPG_teaser_720p.jpg)

We propose a solution to importance sampling specular chains with seed placement using importance probability distributions reconstructed from historical samples. 
This outperforms state-of-the-art unbiased solutions with up to 40× variance reduction.

## Links

[Project Page](https://zhiminfan.work/manifoldPG.html)

[Paper (full)](https://zhiminfan.work/paper/ManifoldPG_Sept28.pdf)

[Paper (compressed)](https://zhiminfan.work/paper/ManifoldPG_Sept23_small.pdf)

Code will be available soon.

## Abstract

Complex visual effects such as caustics are often produced by light paths containing multiple consecutive specular vertices (dubbed specular chains), which pose a challenge to unbiased estimation in Monte Carlo rendering. In this work, we study the light transport behavior within a sub-path that is comprised of a specular chain and two non-specular separators. We show that the specular manifolds formed by all the sub-paths could be exploited to provide coherence among sub-paths. By reconstructing continuous energy distributions from historical and coherent sub-paths, seed chains can be generated in the context of importance sampling and converge to admissible chains through manifold walks. We verify that importance sampling the seed chain in the continuous space reaches the goal of importance sampling the discrete admissible specular chain. Based on these observations and theoretical analyses, a progressive pipeline, manifold path guiding, is designed and implemented to importance sample challenging paths featuring long specular chains. To our best knowledge, this is the first general framework for importance sampling discrete specular chains in regular Monte Carlo rendering. Extensive experiments demonstrate that our method outperforms state-of-the-art unbiased solutions with up to 40× variance reduction, especially in typical scenes containing long specular chains and complex visibility.

