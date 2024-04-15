# CellGPR

We have developed a comprehensive method for automatically integrating functional imaging and structural imaging information at the single neuron level. By inputting data from two-photon calcium imaging and structural slice imaging, the system automatically generates and displays neuronal pairing results. Nine structural slice imaging datasets used for registration and verification are provided, all acquired using the HD-fMOST system with a resolution of 0.65×0.65×2μm³; the functional imaging data comes from two-photon calcium imaging, with a resolution of 1×1μm², covering 22 imaging sites located in layers L2/3 and L5 of the visual cortex. The code for automatic neuronal matching is also provided.

## Dataset

We provide datasets comprising 9 structural slice imaging datasets and data from 22 two-photon imaging sites. You can download the datasets from the following link: [Download Datasets](http://atlas.brainsmatics.org/a/li2404).

### Directory Structure

- **2p/**: Contains two-photon imaging data and dynamically labeled cells, useful for functional analysis.
- **fMOST/**: Contains structural slice imaging data, critical for anatomical insights.
- **results/**: Contains outputs from our scripts, which display neuronal pairing relationships using connecting lines to illustrate possible neuronal connections.

## Compilation Instructions

To compile the matching computation code written in C++, convert the CPP files into a dynamic link library by following these steps:

```bash
cd random_matching/cpp
g++ -o t3.so -shared -fPIC tensor_matching_v3.cpp
```

## Running the Code

1. Place the data files into the `data` folder.
2. Modify the experimental parameters in the `parameter.py` file to match the names used in the data folder.
3. Execute the script by running:

```bash
python neuron_matching.py
```

Ensure all steps are followed accurately to maintain functionality and performance of the processing script.
