

class Network(object):
    """
    """
    def __init__(self, 
                 num_layers, 
                 alg_choice, 
                 alg_params, 
                 num_nodes_per_layer, 
                 cifar_stat, 
                 patch_mode='Adjacent', 
                 image_type='Color'):

        """
        Initialize DeSTIN

        @param num_layers: number of layers 
        @param alg_choice: choice of algorithm - one from ["Clustering", "Auto-Encoder", "LogRegresion"]
        @param alg_params: Dictionary of various parameters needed for learning algorithm 
        @param num_nodes_per_layer: number of nodes present in each layer of DeSTIN
        @param cifar_stat: parameters needed for the CIFAR dataset
        @param patch_mode: patch selection mode for images 
        @param image_type: type of images to be used
        """

        self.network_belief = {}
        self.lowest_layer = 1
        # this is going to store beliefs for every image DeSTIN sees
        self.network_belief['belief'] = np.array([])
        self.save_belief_option = 'True'
        self.belief_file_name = 'beliefs.mat'
        self.number_of_layers = num_layers
        self.algorithm_choice = alg_choice
        self.algorithm_params = alg_params
        self.number_of_nodesPerLayer = num_nodes_per_layer
        self.patch_mode = patch_mode
        self.image_type = image_type
        self.layers = [
            [Layer(j, num_nodes_per_layer[j], cifar_stat, self.patch_mode, self.image_type) for j in range(num_layers)]]