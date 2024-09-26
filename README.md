# Demo and replication code for: **Spatial embedding promotes a specific form of modularity with low entropy and heterogeneous spectral dynamics**

Cornelia Sheeran* [1,2], Andrew S. Ham* [1,3], Duncan E. Astle [1,4], 
Jascha Achterberg+ [1,5], Danyal Akarca+ [1,6,7]
  
1.	MRC Cognition and Brain Sciences Unit, University of Cambridge, UK
2.	Center for the Study of Complex Systems, University of Michigan, USA
3.	Harvard Medical School, Harvard University, USA
4.	Department of Psychiatry, University of Cambridge, UK
5.	Department of Physiology, Anatomy and Genetics, University of Oxford, UK
6.	Department of Electrical and Electronic Engineering, Imperial College London, UK
7.	I-X, Imperial College London, UK

* Co-lead authors, + Co-lead senior authors, Corresponding author: Danyal Akarca (d.akarca@imperial.ac.uk)

For the corresponding data, please see: https://osf.io/vw2ac/

<img width="1360" alt="figure_1" src="https://github.com/user-attachments/assets/92631003-68d4-4400-94f1-23504f5a8ad4">

## Contents

- seRSNN_demo.ipynb - notebook demo for training spatially embedded spiking RNNs using snnTorch


- rate_data.mat: contains rate RNNs over training across network types (see Achterberg & Akarca, et al. 2023 https://github.com/8erberg/spatially-embedded-RNN for more detail)
- rate_generate_statistics.m: code for generating relevant entropic measures of the weights and eigenspectrum for rate networks (rate_rnns_entropy_statistics.mat is the output)
- rate_figures.m - code for generating all rate RNN figures


- spiking_data.mat: contains spiking RNNs over training
- spiking_generate_statistics.m: code for generating relevant entropic measures of the weights and eigenspectrum for spiking networks (spiking_rnns_entropy_statistics.mat is the output)
- spiking_figures.m - code for generating all spiking RNN figures
