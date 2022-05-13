import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import umap.umap_ as umap
from sklearn.metrics import accuracy_score, rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import torch
from torch import nn, optim
import auto_encoder


RANDOM_SEED = 0


def get_data(train_csv, test_csv):
    train_df = pd.read_csv(train_csv, index_col=0)
    test_df = pd.read_csv(test_csv, index_col=0)
    x_train = train_df.iloc[:, :-2]
    y_train = train_df[['pop', 'super_pop']]
    x_test = test_df.iloc[:, :-2]
    y_test = test_df[['pop', 'super_pop']]
    return x_train, y_train, x_test, y_test


def plot_eigenvalues(data_df):
    pca = PCA(random_state=RANDOM_SEED).fit(data_df)
    eigenvalues = pca.explained_variance_
    fig = px.line(x=range(1, len(eigenvalues)+1), y=eigenvalues)
    fig.show()


def apply_pca(x_train, x_test, pca_shape):
    np.random.seed(RANDOM_SEED)
    pca = PCA(n_components=pca_shape, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)
    if x_test is None:
        return x_train_pca
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_test_pca


def apply_umap(x_train, x_test, umap_shape):
    reducer = umap.UMAP(n_components=umap_shape, random_state=RANDOM_SEED)
    x_train_umap = reducer.fit_transform(x_train)
    if x_test is None:
        return x_train_umap
    x_test_umap = reducer.transform(x_test)
    return x_train_umap, x_test_umap


def ae(x_train, x_test, pca_shape, enc_shape):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    pca = PCA(n_components=pca_shape, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    ae = auto_encoder.AutoEncoder(in_shape=pca_shape, enc_shape=enc_shape).double().to(device)
    x_train_pca = torch.from_numpy(x_train_pca).to(device)
    error = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters())
    auto_encoder.train(ae, error, optimizer, 1000, x_train_pca)

    x_train_ae = ae.encode(x_train_pca).cpu().detach().numpy()
    if x_test is None:
        return x_train_ae
    x_test_pca = torch.from_numpy(pca.transform(x_test)).to(device)
    x_test_ae = ae.encode(x_test_pca).cpu().detach().numpy()
    return x_train_ae, x_test_ae


def cae(x_train, x_test, pca_shape, enc_shape):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    pca = PCA(n_components=pca_shape, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    cae = auto_encoder.ConvolutionalAutoEncoder(in_shape=pca_shape, enc_shape=enc_shape).double().to(device)
    x_train_pca = torch.from_numpy(x_train_pca).to(device).unsqueeze(1)
    error = nn.MSELoss()
    optimizer = optim.Adam(cae.parameters())
    auto_encoder.train(cae, error, optimizer, 1000, x_train_pca)

    x_train_cae = cae.encode(x_train_pca).cpu().detach().numpy().squeeze()
    if x_test is None:
        return x_train_cae
    x_test_pca = torch.from_numpy(pca.transform(x_test)).to(device).unsqueeze(1)
    x_test_cae = cae.encode(x_test_pca).cpu().detach().numpy().squeeze()
    return x_train_cae, x_test_cae


def lstmae(x_train, x_test, pca_shape, enc_shape):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    pca = PCA(n_components=pca_shape, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    lstmae = auto_encoder.LSTMAutoEncoder(in_shape=pca_shape, enc_shape=enc_shape).double().to(device)
    x_train_pca = torch.from_numpy(x_train_pca).to(device)
    error = nn.MSELoss()
    optimizer = optim.Adam(lstmae.parameters())
    auto_encoder.train(lstmae, error, optimizer, 1000, x_train_pca)

    x_train_lstmae = lstmae.encode(x_train_pca).cpu().detach().numpy()
    if x_test is None:
        return x_train_lstmae
    x_test_pca = torch.from_numpy(pca.transform(x_test)).to(device)
    x_test_lstmae = lstmae.encode(x_test_pca).cpu().detach().numpy()
    return x_train_lstmae, x_test_lstmae


def gruae(x_train, x_test, pca_shape, enc_shape):
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    pca = PCA(n_components=pca_shape, random_state=RANDOM_SEED)
    x_train_pca = pca.fit_transform(x_train)

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    gruae = auto_encoder.GRUAutoEncoder(in_shape=pca_shape, enc_shape=enc_shape).double().to(device)
    x_train_pca = torch.from_numpy(x_train_pca).to(device)
    error = nn.MSELoss()
    optimizer = optim.Adam(gruae.parameters())
    auto_encoder.train(gruae, error, optimizer, 1000, x_train_pca)

    x_train_gruae = gruae.encode(x_train_pca).cpu().detach().numpy()
    if x_test is None:
        return x_train_gruae
    x_test_pca = torch.from_numpy(pca.transform(x_test)).to(device)
    x_test_gruae = gruae.encode(x_test_pca).cpu().detach().numpy()
    return x_train_gruae, x_test_gruae


def plot_dimensions(x_train, y_train, x_test, y_test, title):
    labels = ["Component {}".format(x) for x in range(1, x_train.shape[1] + 1)]
    train_label = pd.DataFrame(x_train, columns=labels).join(y_train.reset_index(drop=True), how='outer').sort_values(by="Population")
    fig = px.scatter(train_label, x=labels[0], y=labels[1], color="Population", title=title)
    fig.show()

    if x_test is not None:
        test_label = pd.DataFrame(x_test, columns=labels).join(y_test.reset_index(drop=True), how='outer').sort_values(by="Population")
        fig = px.scatter(test_label, x=labels[0], y=labels[1], color="Population", title=title)
        fig.show()


def linear_regression(x_train, y_train, x_test, y_test):
    np.random.seed(RANDOM_SEED)
    model = LogisticRegression().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def random_forest(x_train, y_train, x_test, y_test):
    np.random.seed(RANDOM_SEED)
    model = RandomForestClassifier().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def svm(x_train, y_train, x_test, y_test):
    np.random.seed(RANDOM_SEED)
    model = SVC().fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def mlp(x_train, y_train, x_test, y_test):
    np.random.seed(RANDOM_SEED)
    model = MLPClassifier(max_iter=500).fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def k_means(x, y):
    np.random.seed(RANDOM_SEED)
    model = KMeans(n_clusters=5).fit(x)
    y_pred = model.labels_
    return rand_score(y, y_pred), pd.DataFrame(y_pred, columns=['Population'])['Population'].astype(str)


def run_inference(x_train, x_test, y_train, y_test, model, n_dimensions):
    print("Testing {}...".format(model))
    print("Logistic Regression:", linear_regression(x_train, y_train, x_test, y_test))
    print("Random Forest:", random_forest(x_train, y_train, x_test, y_test))
    print("SVM:", svm(x_train, y_train, x_test, y_test))
    print("MLP:", mlp(x_train, y_train, x_test, y_test))
    plot_dimensions(x_train, y_train, x_test, y_test, title="Populations After Running {} to {} Dimensions".format(model, n_dimensions))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_data(train_csv='./data/chr1_train.csv', test_csv='./data/chr1_test.csv')
    # # plot_eigenvalues(x_train)
    geo = 'super_pop'
    y_train = y_train[geo].rename("Population")
    y_test = y_test[geo].rename("Population")
    x = np.concatenate((x_train, x_test), axis=0)
    y = pd.concat([y_train, y_test])
    n_dimensions = 2

    model = "PCA"
    print("Training {}...".format(model))
    x_train_pca, x_test_pca = apply_pca(x_train, x_test, pca_shape=n_dimensions)
    run_inference(x_train_pca, x_test_pca, y_train, y_test, model, n_dimensions)

    model = "UMAP"
    print("Training {}...".format(model))
    x_train_umap, x_test_umap = apply_umap(x_train, x_test, umap_shape=n_dimensions)
    run_inference(x_train_umap, x_test_umap, y_train, y_test, model, n_dimensions)

    model = "AE"
    print("Training {}...".format(model))
    x_train_ae, x_test_ae = ae(x_train, x_test, pca_shape=256, enc_shape=n_dimensions)
    run_inference(x_train_ae, x_test_ae, y_train, y_test, model, n_dimensions)

    model = "CAE"
    print("Training {}...".format(model))
    x_train_cae, x_test_cae = cae(x_train, x_test, pca_shape=256, enc_shape=n_dimensions)
    run_inference(x_train_cae, x_test_cae, y_train, y_test, model, n_dimensions)

    model = "LSTMAE"
    print("Training {}...".format(model))
    x_train_lstmae, x_test_lstmae = lstmae(x_train, x_test, pca_shape=256, enc_shape=n_dimensions)
    run_inference(x_train_lstmae, x_test_lstmae, y_train, y_test, model, n_dimensions)

    model = "GRUAE"
    print("Training {}...".format(model))
    x_train_gruae, x_test_gruae = gruae(x_train, x_test, pca_shape=256, enc_shape=n_dimensions)
    run_inference(x_train_gruae, x_test_gruae, y_train, y_test, model, n_dimensions)
