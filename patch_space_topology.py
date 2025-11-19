import numpy as np
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
from sklearn.feature_extraction.image import extract_patches_2d


def extract_patches(img, kernel_size, stride, padding):
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=-1)
    img_padded = np.pad(img, [(padding, padding), (padding, padding), (0, 0)], mode='constant')
    patches = extract_patches_2d(img_padded, (kernel_size, kernel_size), max_patches=None, random_state=None)
    patches = patches[::stride, :, :]  # applying stride
    flattened_patches = patches.reshape(patches.shape[0], -1)
    return flattened_patches


def calculate_persistent_entropy(patches, n_samples=None):
    # Randomly select a subset of patches if required
    if n_samples is not None:
        np.random.shuffle(patches)
        sample_patches = patches[:n_samples]
    else:
        sample_patches = patches
    
    # Create Vietoris-Rips complex
    VR = VietorisRipsPersistence()
    diagrams = VR.fit_transform([sample_patches])
    
    n_samples_distinct = np.sum([i[-1] == 0 for i in diagrams[0]])
    
    # Calculate persistent entropy
    PE = PersistenceEntropy(normalize=False, nan_fill_value = np.log2(min(n_samples_distinct, len(sample_patches)))+1e-5)
    entropy = PE.fit_transform(diagrams)/(np.log2(min(n_samples_distinct, len(sample_patches)))+1e-5)
    
    return entropy


def measure_entropy(img, kernel_size, stride, padding, n_samples=None):
    
    patches = extract_patches(img, kernel_size, stride, padding)
    entropy = calculate_persistent_entropy(patches, n_samples)
    
    return entropy


class FeatureExtractor():
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.outputs = {}
        
        self.hooks = {}
        
        # Создаем хуки для нужных слоев
        for name, module in self.model.named_modules():
            if name in self.layer_names:
                self.hooks[name] = module.register_forward_hook(self.named_hook__(name))
    
    def named_hook__(self, module_name):
        def _hook(module, input, output):
            self.outputs[module_name] = output.detach().clone()
        
        return _hook
    
    def extract_features(self, x):
        _ = self.model(x)
        res = {i: j for i,j in self.outputs.copy().items()}
        self.clear_outputs()
        return res
    
    def clear_outputs(self):
        """Очищает внутренние буферы с выходными данными."""
        self.outputs.clear()


class EntropyMeasurer():
    def __init__(self, model, entropy_params_list):
        self.model = model
        self.entropy_params_list = entropy_params_list
        
        self.feature_extractor = FeatureExtractor(model, [i for i in entropy_params_list.keys() if i != "input"])
    
    
    def measure_entropies(self, x):
        res = {}
        
        res["input"] = measure_entropy(x.squeeze(0).movedim(0,-1), **self.entropy_params_list["input"])
        
        model_embeddings = self.feature_extractor.extract_features(x)
        
        for key in [i for i in self.entropy_params_list.keys() if i != "input"]:
            res[key] = measure_entropy(model_embeddings[key].squeeze(0).movedim(0,-1), **self.entropy_params_list[key])
        
        return res
