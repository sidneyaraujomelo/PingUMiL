import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data_df = pd.read_csv("pingumil/experiments/sswinpred/embeddings/embedding_dict.csv", index_col=0)
#print(data_df)

player1_embs = None
player2_embs = None
with open("pingumil/experiments/sswinpred/embeddings/player1embs.pt", "rb") as fp:
    player1_embs = torch.load(fp)
with open("pingumil/experiments/sswinpred/embeddings/player2embs.pt", "rb") as fp:
    player2_embs = torch.load(fp)
    
assert len(data_df.index) == player1_embs.shape[0]
assert len(data_df.index) == player2_embs.shape[0]

all_embs = torch.concat((player1_embs, player2_embs)).cpu()
print(all_embs.shape)

def get_kmeans(embeds, n):
    best_km = None
    best_labels_score = -1
    patience = 25
    for k in range(100):
        if k%10==0:
            print(f"Current step: {k}")
        km = KMeans(n_clusters=n).fit(embeds)
        score = silhouette_score(embeds, km.labels_, sample_size=1000)
        if score > best_labels_score:
            best_labels_score = score
            best_km = km
            patience = 10
        else:
            patience = patience - 1
        if patience == 0:
            break
    return best_km, best_labels_score

def run_kmeans(all_embs, k=5, output_prefix="player_profiles_sswinpred"):
    km, best_km_score = get_kmeans(all_embs, k)

    player1_labels = km.predict(player1_embs.cpu())
    print(player1_labels)
    unique, counts = np.unique(player1_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    player2_labels = km.predict(player2_embs.cpu())
    print(player2_labels)
    unique, counts = np.unique(player2_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    kmeans_datadf = data_df.copy()
    print(kmeans_datadf)
    kmeans_datadf["player01_cluster"] = player1_labels
    kmeans_datadf["player02_cluster"] = player2_labels

    print(kmeans_datadf)
    kmeans_datadf.to_csv(f"{output_prefix}_kmeans{k}.csv")
    
def get_spectral(embeds, n, gamma=0.55):
    spectral = SpectralClustering(n_clusters=n, n_init=20, gamma=gamma, assign_labels='discretize').fit(embeds)
    return spectral

def run_spectral(all_embs, k=5, output_prefix="player_profiles_sswinpred"):
    spectral = get_spectral(all_embs, k)
    
    unique, counts = np.unique(spectral.labels_, return_counts=True)
    print(dict(zip(unique, counts)))
    
    player1_labels = spectral.labels_[:len(player1_embs.cpu())]
    print(player1_labels)
    unique, counts = np.unique(player1_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    player2_labels = spectral.labels_[len(player1_embs.cpu()):]
    print(player2_labels)
    unique, counts = np.unique(player2_labels, return_counts=True)
    print(dict(zip(unique, counts)))

    spectral_datadf = data_df.copy()
    print(spectral_datadf)
    spectral_datadf["player01_cluster"] = player1_labels
    spectral_datadf["player02_cluster"] = player2_labels

    spectral_datadf.to_csv(f"{output_prefix}_spectralg55k{k}.csv")
    
if __name__ == "__main__":
    output_prefix="player_profiles_sswinpredllr"
    for k in [5,10,20]:
        run_spectral(all_embs, k, output_prefix=output_prefix)
        run_kmeans(all_embs, k, output_prefix=output_prefix)