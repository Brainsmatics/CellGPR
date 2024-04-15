# CellGPR
We have developed a comprehensive method for automatically integrating functional imaging and structural imaging information at the single neuron level. By inputting data from two-photon calcium imaging and structural slice imaging, the system automatically generates and displays neuronal pairing results. Nine structural slice imaging datasets used for registration and verification are provided, all acquired using the HD-fMOST system with a resolution of 0.65×0.65×2μm³; the functional imaging data comes from two-photon calcium imaging, with a resolution of 1×1μm², covering 22 imaging sites located in layers L2/3 and L5 of the visual cortex. The code for automatic neuronal matching is also provided.
# Datasets
We provide data on imaging of 9 structural slices and 22 two-photon imaging sites. The download link for the data can be found at: http://atlas.brainsmatics.org/a/li2404 In the dataset, the 2p folder corresponds to dynamic cells for two-photon imaging and labeling, the fMOST file corresponds to structural slice imaging data, and the results contain program generated results. The pairing relationship of neurons will be displayed in the form of lines.
# Compile
Compile the CPP file in the matching calculation into a dynamically linked file
'''
cd random_matching/cpp
g++ -o t3.so -shared -fPIC tensor_matching_v3.cpp
'''
# Run code
Place the data file in the data folder
Modify the parameters of the experiment in the parameter.py folder to match the name of the data in the data section.
And run the code
'python neuron_matching.py'
