# Heterogeneous Virtual Growth Algorithm

[![DOI](https://zenodo.org/badge/785450642.svg)](https://zenodo.org/doi/10.5281/zenodo.10963129)

This program generates patterns with seamlessly connected building blocks. 

The program generalizes the [virtual growth scheme](https://www.science.org/doi/full/10.1126/science.abn1459) to account for 
spatially varying frequency hints. It also converts the generated patterns to
FEA meshes and evaluate their [homogenized elasticity matrices](https://www.sciencedirect.com/science/article/pii/S0045782512000941).

## Examples

2D pattern

<img src="images/2d/symbolic_graph.jpg" width="220" height="220">

3D pattern

<img src="images/3d/symbolic_graph.jpg" width="220" height="220">

2D FEA mesh (Q4 elements)

<img src="images/2d/fem_mesh.jpg" width="220" height="220">

3D FEA mesh (frame elements)

<img src="images/3d/fem_mesh.jpg" width="220" height="220">

## How to use this program

### Installation
To download the code, run the following command in your terminal:
```
git clone https://github.com/jiayingqi/Heterogeneous-Virtual-Growth
```

Then run the following command to install xvfb for visualization:
```
apt-get -qq update && apt-get -y install libgl1-mesa-dev xvfb
```

Finally, run the following command to install the required Python packages:
```
pip3 install -r requirements.txt
```

### Start with the program
Generate the adjacency rules:
```
python3 scripts/generate_rules_script.py
```

Generate the patterns:
```
python3 scripts/virtual_growth_script.py
```

Convert the patterns to FEA meshes:
```
python3 scripts/generate_mesh_script.py
```

Evaluate the homogenized elasticity matrices:
```
python3 scripts/homogenize_script.py
```

## Authors, sponsors, and citation

### Authors
- Yingqi Jia (yingqij2@illinois.edu)
- Ke Liu (liuke@pku.edu.cn)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

### Sponsor
- David C. Crawford Faculty Scholar Award from the Department of Civil and
  Environmental Engineering and Grainger College of Engineering at the
  University of Illinois

### Citations
- Jia, Y., Liu, K., Zhang, X.S., 2024. Modulate stress distribution with
  bio-inspired irregular architected materials towards optimal tissue support.
  Nature Communications 15, 4072. https://doi.org/10.1038/s41467-024-47831-2
- Jia, Y., Liu, K., Zhang, X.S., 2024. Topology optimization of irregular
  multiscale structures with tunable responses using a virtual growth rule.
  Computer Methods in Applied Mechanics and Engineering 425, 116864.
  https://doi.org/10.1016/j.cma.2024.116864
