import pandas as pd
from sklearn.decomposition import PCA

DEFAULT_SIZE = 100


def get_image(name, size=DEFAULT_SIZE):
    return f'<img src="images/{name}.jpg" width={size} />'


def get_images(*names, size=DEFAULT_SIZE):
    return [get_image(name, size=size) for name in names]

def get_pca(n_components, df):
    _df = pd.DataFrame(df)
    pca = PCA(n_components=n_components)
    pca.fit(_df)
    return pd.DataFrame(pca.transform(_df), columns=[f'PCA {i+1}'for i in range(n_components)], index=df.index)
