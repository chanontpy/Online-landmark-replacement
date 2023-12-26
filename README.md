# Online-landmark-replacement

- This repository provides the code for the article [Online landmark replacement for out-of-sample dimensionality reduction methods](https://arxiv.org/abs/2311.12646).

- The main file is `S-curve_landmark_replacement_embedding.py` which contains landmark replacement algorithm (Algorithm 2) and the two diemensional embedding of S-curve data. The S-curve data set is artificial and is provided by `sklearn` package. Note that one needs to import two assisting .py files: `Online_LandmarkReplacement_Functions.py` and `LandmarkMDS.py`.

- The code with which `S-curve_landmark_replacement_embedding.py` is directly applicable to other data sets.

## Online_LandmarkReplacement_Functions.py
_ Importing and executing the assisting files
  _ `from Online_LandmarkReplacement_Functions import *`
  _ `from LandmarkMDS import *`

_ Some Variables' description
  _ `L` is the set of landmarks.
  _ `m` is the (permitted) number of landmarks.
  _ `epsilon` is the parameter determining the presence of edges in a geometric graph.
  _ `sq_distance_mat` is the matrix encoding squared Euclidean distances of pairs of data points.
  _ `Landmark_eigenpair` contains the eigenpair(eigenvalues and unit eigenvectors) of the double mean-centered `sq_distance_mat`.
  _ `k` is the embedding dimension, which, in this case, equals to $2$.
  _ `L_prime` is the matrix $L'_k$ in Landmark MDS.
  _ `emned_coord` is the list containing the two dimensional embedding coordinates, where line 115 is the Landmark MDS map.
  
_ Summary of the code
  _ Lines 5-13 create S-curve using `sklearn.datasets.make_s_curve` package.
  _ Lines 15-30 are for constructing a geometric graph.
  _ The landmark replacement algorithm corresponds to lines 43-79.
  _ The embedding by Landmark multidimensional scaling corresponds to lines 81-121.

## LandmarkMDS.py
- The code for tie-decay network is under `if __name__ == '__main__'`.
