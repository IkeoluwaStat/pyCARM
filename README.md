# pyCARM - Cellular Automata for Aircraft Arrival Modeling

# Project

## fGP-DRT: finite Gaussian Process Distribution of Relaxation Times

This repository contains some of the source code used for the paper titled *The Probabilistic Deconvolution of the Distribution of Relaxation Times with Finite Gaussian Processes*. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119. The article is available online at [Link](https://doi.org/10.1016/j.electacta.2022.140119) and in the [docs](docs) folder. 

# Introduction
Electrochemical impedance spectroscopy (EIS) is a tool widely used to study the properties of electrochemical systems. The distribution of relaxation times (DRT) is a widely used approach, in electrochemistry, biology and material science, for the analysis of electrochemical impedance 
spectroscopy (EIS) data [1]. Deconvolving the DRT from EIS data is quite challenging because an ill-posed problem needs to be solved [2-5]. Several approaches such as ridge regression, ridge and lasso regression, Bayessian and hierarchical Bayesian, Hilbert transform and Gaussian process methods have been used [2-7]. Gaussian processes can be used to regress EIS data, quantify uncertainty, and deconvolve the DRT. However, previously developed DRT models based on Gaussian processes do not constrain the DRT to be non-negative and can only use the imaginary part of EIS spectra [8,9]. Therefore, we overcome both issues by using a finite Gaussian process approximation to develop a new framework called the finite Gaussian process distribution of relaxation times (fGP-DRT) [10]. The analysis on artificial EIS data shows that the fGP-DRT method consistently recovers exact DRT from noise-corrupted EIS spectra while accurately regressing experimental data. Furthermore, the fGP-DRT framework is used as a machine learning tool to provide probabilistic estimates of the impedance at unmeasured frequencies. The method is further validated against experimental data from fuel cells and batteries. In short, this work develops a novel probabilistic approach for the analysis of EIS data based on Gaussian process, opening a new stream of research for the deconvolution of DRT. 

![Screenshot 2022-02-12 165048](https://user-images.githubusercontent.com/99115272/153704506-9184e95d-4a07-4233-ac7f-cbb4bbdee680.gif)

# Dependencies
numpy

scipy

matplotlib

pandas

# Tutorials
1. **ex1_single ZARC Model.ipynb**: this notebook gives detail procedure of how to recover the DRT from the impedance generated using a single ZARC model consisting of a resistance placed in parallel to a constant phase element (CPE) The frequency range is from 1E-4 Hz to 1E4 Hz with 10 points per decade (ppd).
2. **ex2_double ZARC Model.ipynb** : this notebook demonstrates how the fGP-DRT can capture overlapping timescales with two ZARC models arranged in series. The frequency range is from 1E-4 Hz to 1E4 Hz with 10 ppd.
3. **ex3_single_ZARC_plus_an_inductor.pynb** : this notebook adds an inductor to the model used in "**example1_single ZARC Model.ipynb**"
 

# Citation

```
@article{ogedengbe2025cellular,
  title={Cellular automata for the investigation of navigation dynamics and aircraft mix in terminal arrival traffic},
  author={Ogedengbe, Ikeoluwa Ireoluwa and Tai, Tak Shing and Wong, KY Michael and Liem, Rhea P},
  journal={Physica A: Statistical Mechanics and its Applications},
  pages={130628},
  year={2025},
  publisher={Elsevier}
}

```

# References
[1] Maradesa, A., Py, B., Quattrocchi, E., & Ciucci, F. (2022). The probabilistic deconvolution of the distribution of relaxation times with finite Gaussian processes. Electrochimica Acta, 413, 140119. https://doi.org/10.1016/j.electacta.2022.140119.

[2] Ciucci, F. (2018). Modeling electrochemical impedance spectroscopy. Current Opinion in Electrochemistry.132-139. https://doi.org/10.1016/j.coelec.2018.12.003. 

