import numpy as np
from sksfa import SFA, HSFA

def load_data():
    path = 'data/data_squareRoom.npy'
    data = np.load(path)

    # Print or inspect the data
    print(data)
    print("Data shape:", data.shape)
    return data

def crop_images(data):
    crop_h, crop_w, dim_step = 20, 10, 1
    data = data[:, crop_h:-crop_h, crop_w:-crop_w][:, ::dim_step][:, :, ::dim_step]
    return data

def normalize(data):
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return data

def sfa(data):
    reshaped_data = data.reshape(data.shape[0], -1)

    # Create and train an SFA node
    sfa = SFA(n_components=1000)
    features = sfa.fit_transform(reshaped_data)

    # Extract slow features
    print("Extracted slow features shape:", features.shape)
    return features

def hsfa(data):
    layer_configurations = [(5, 19, 5, 19, 1, 1)]
    input_shape=(data.shape[1], data.shape[2], data.shape[3])
    hsfa = HSFA(n_components=1000, input_shape=input_shape, layer_configurations=layer_configurations, verbose=True)
    hsfa = hsfa.fit(data)
    features = hsfa.transform(data)

    print("Extracted slow features shape:", features.shape)
    return features

if __name__ == '__main__':
    data = load_data()
    cropped_images = crop_images(data)
    print(cropped_images.shape)
    normalized_image = normalize(cropped_images)
    h_features = hsfa(normalized_image)
    features = sfa(normalized_image)

