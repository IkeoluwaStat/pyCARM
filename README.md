# pyCARM - Cellular Automata for Aircraft Arrival Modeling

# Project

## fGP-DRT: finite Gaussian Process Distribution of Relaxation Times

This repository contains some of the source code used for the paper titled *Cellular automata for the investigation of navigation dynamics and aircraft mix in terminal arrival traffic*. Physica A, 671 (2025), 130628. https://doi.org/10.1016/j.physa.2025.130628. The article is available online at [Link](https://doi.org/10.1016/j.physa.2025.130628) and in the [docs](docs) folder. 

# Introduction


<img width="607" height="370" alt="Image" src="https://github.com/user-attachments/assets/f73621d4-d229-4439-82db-e0015056b3c7" />

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

