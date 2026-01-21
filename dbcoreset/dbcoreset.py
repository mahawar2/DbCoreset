import numpy as np
from sklearn.cluster import DBSCAN




class DbCoreset:
def __init__(self, budget, eps=0.005, minpts=10, lambda1=0.7, lambda2=0.3):
self.budget = budget
self.eps = eps
self.minpts = minpts
self.lambda1 = lambda1
self.lambda2 = lambda2


def squared_euclidean(self, a, b):
return np.sum((a - b) ** 2)


def run(self, F):
"""
F: (N, D) feature matrix
Returns selected indices
"""
N = F.shape[0]
dbscan = DBSCAN(eps=self.eps, min_samples=self.minpts)
labels = dbscan.fit_predict(F)


clusters = {}
outliers = []


for i, lbl in enumerate(labels):
if lbl == -1:
outliers.append(i)
else:
clusters.setdefault(lbl, []).append(i)


Z = []
Z.extend(outliers)


N_prime = self.budget - len(outliers)
k = len(clusters)


per_cluster = max(1, N_prime // k) if k > 0 else 0


for _, idxs in clusters.items():
if len(idxs) <= per_cluster:
Z.extend(idxs)
N_prime -= len(idxs)
else:
scores = []
mu = None
s_sum = 0


for i, idx in enumerate(idxs):
f = F[idx]
if i == 0:
s_i = 1
mu = f
else:
phi = self.squared_euclidean(f, mu)
s_i = self.lambda1 * phi + self.lambda2 / i
mu = (mu * i + f) / (i + 1)


s_sum += s_i
p_i = s_i / s_sum
scores.append((p_i, idx))


scores.sort(reverse=True, key=lambda x: x[0])
selected = [idx for _, idx in scores[:per_cluster]]
Z.extend(selected)
N_prime -= per_cluster


return np.array(Z)
