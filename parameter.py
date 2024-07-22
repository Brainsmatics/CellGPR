# This script provides parameters for the registration of functional imaging and structural imaging.

# Path to the dataset
data_path = './data/202277'

# Parameters for detecting neurons in HD-fMOST three-dimensional structural slice imaging
conf_theta = 0.1  # Confidence threshold for cell recognition
distance_thread = 8  # Minimum distance between two cells
batch_size = 2  # Batch size for predictions
model_path = './Neuron_Detection/model.pth'  # Path to load the trained model

# Parameters for matching
thickness_L23 = 30  # Thickness for two-photon imaging of Layer 2/3, default is 30μmc
thickness_L5 = 70  # Thickness for two-photon imaging of Layer 5, default is 70μm
theta = 1e-2  # Exponential parameter for evaluating hyperedge similarity
step0 = 10  # Step size in iterative calculations
dll_path = 'C:/Program Files/Git/mingw64/bin' # Path to the DLL file 
