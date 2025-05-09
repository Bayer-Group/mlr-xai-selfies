import numpy as np

def get_features(data, CLOUMS):
    """
    Preprocesses the features according to theyr type.

    Keyword arguments:
    -- data: Table where the features are stored.
    -- CLOUMS: Colum where the feature is stored.

    Returns:
    -- X: Preprocessed feature.
    """
    features = []

    for i in CLOUMS:
        if type(data[i].values[0]) is np.ndarray:
            x = np.stack(data[i].values)
        else:
            x = data[i].values.reshape(-1, 1)
        features.append(x)

    X = np.concatenate(features, axis=1)
    return X