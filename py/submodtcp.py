import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class FacilityLocation:
    def __init__(self,
                 source_embeddings=None,
                 target_embeddings=None,
                 metric='cosine',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 similarity_matrix=None):
        self.device = device
        self.metric = metric

        # Option 1: Use pre-computed similarity matrix
        if similarity_matrix is not None:
            self.similarity_matrix = torch.tensor(similarity_matrix, device=self.device, dtype=torch.float32)
            self.n = similarity_matrix.shape[0]
            self.source_embeddings = None
        # Option 2: Compute from embeddings
        elif source_embeddings is not None:
            self.source_embeddings = source_embeddings
            self.n = source_embeddings.shape[0]

            if target_embeddings is None:
                target_embeddings = source_embeddings

            # Compute similarity matrix based on metric
            if metric == 'cosine':
                similarity = cosine_similarity(target_embeddings, source_embeddings)
            elif metric == 'euclidean':
                # Convert Euclidean distance to similarity: smaller distance = higher similarity
                distances = euclidean_distances(target_embeddings, source_embeddings)
                # Use 1/(1+distance) to convert to similarity in [0, 1] range
                similarity = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

            self.similarity_matrix = torch.tensor(similarity, device=self.device, dtype=torch.float32)
        else:
            raise ValueError("Must provide either source_embeddings or similarity_matrix")

    def evaluate(self, subset):
        if not subset:
            return 0.0
        subset_tensor = torch.tensor(subset, device=self.device)
        return torch.max(self.similarity_matrix[:, subset_tensor], dim=1)[0].sum().cpu().numpy()

    def marginal_gain(self, subset, candidate):
        if not subset:
            return self.similarity_matrix[:, candidate].sum()

        subset_tensor = torch.tensor(subset, device=self.device)
        current_max = torch.max(self.similarity_matrix[:, subset_tensor], dim=1)[0]
        new_max = torch.max(current_max, self.similarity_matrix[:, candidate])
        return (new_max - current_max).sum()

    def lazy_greedy(self, budget):
        subset = []
        gains = torch.zeros(self.n, device=self.device)
        gain_values = []

        for i in range(self.n):
            gains[i] = self.marginal_gain(subset, i)

        for _ in tqdm(range(budget), desc="FL Selection"):
            while True:
                candidate = int(torch.argmax(gains).cpu().numpy())
                exact_gain = self.marginal_gain(subset, candidate)

                if exact_gain >= torch.max(gains) - 1e-6:
                    subset.append(candidate)
                    gain_values.append((candidate, float(exact_gain.cpu().numpy())))
                    gains[candidate] = float('-inf')
                    break
                else:
                    gains[candidate] = exact_gain

        return subset, gain_values


class LogDeterminant:
    def __init__(self,
                 source_embeddings=None,
                 target_embeddings=None,
                 lambda_param=1e-6,
                 metric='cosine',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 similarity_matrix=None):
        self.device = device
        self.lambda_param = lambda_param
        self.metric = metric

        # Option 1: Use pre-computed similarity matrix
        if similarity_matrix is not None:
            self.similarity_matrix = torch.tensor(similarity_matrix, device=self.device, dtype=torch.float32)
            self.n = similarity_matrix.shape[0]
            self.source_embeddings = None
        # Option 2: Compute from embeddings
        elif source_embeddings is not None:
            self.source_embeddings = source_embeddings
            self.n = source_embeddings.shape[0]

            if target_embeddings is None:
                target_embeddings = source_embeddings

            # Compute similarity matrix based on metric
            if metric == 'cosine':
                similarity = cosine_similarity(target_embeddings, source_embeddings)
            elif metric == 'euclidean':
                # Convert Euclidean distance to similarity: smaller distance = higher similarity
                distances = euclidean_distances(target_embeddings, source_embeddings)
                # Use 1/(1+distance) to convert to similarity in [0, 1] range
                similarity = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

            self.similarity_matrix = torch.tensor(similarity, device=self.device, dtype=torch.float32)
        else:
            raise ValueError("Must provide either source_embeddings or similarity_matrix")

    def marginal_gain(self, subset, candidate):
        if not subset:
            k_ii = self.similarity_matrix[candidate, candidate]
            return torch.log(k_ii + self.lambda_param)

        subset_tensor = torch.tensor(subset, device=self.device)
        K_S = self.similarity_matrix[subset_tensor][:, subset_tensor]
        K_S_reg = K_S + self.lambda_param * torch.eye(len(subset), device=self.device)
        k_i = self.similarity_matrix[candidate, subset_tensor].unsqueeze(1)
        k_ii = self.similarity_matrix[candidate, candidate]

        try:
            K_S_inv_k_i = torch.linalg.solve(K_S_reg, k_i)
            schur_complement = k_ii + self.lambda_param - torch.dot(k_i.squeeze(), K_S_inv_k_i.squeeze())
            marginal_gain = torch.log(torch.clamp(schur_complement, min=1e-10))
        except:
            marginal_gain = torch.tensor(1e-10, device=self.device)

        return marginal_gain

    def lazy_greedy(self, budget):
        subset = []
        gains = torch.zeros(self.n, device=self.device)
        gain_values = []

        for i in range(self.n):
            gains[i] = self.marginal_gain(subset, i)

        for _ in tqdm(range(budget), desc="LogDet Selection"):
            while True:
                candidate = int(torch.argmax(gains).cpu().numpy())
                exact_gain = self.marginal_gain(subset, candidate)

                if exact_gain >= torch.max(gains) - 1e-6:
                    subset.append(candidate)
                    gain_values.append((candidate, float(exact_gain.cpu().numpy())))
                    gains[candidate] = float('-inf')
                    break
                else:
                    gains[candidate] = exact_gain

        return subset, gain_values


class GraphCut:
    def __init__(self,
                 source_embeddings=None,
                 target_embeddings=None,
                 metric='cosine',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 similarity_matrix=None):
        self.device = device
        self.metric = metric

        # Option 1: Use pre-computed similarity matrix
        if similarity_matrix is not None:
            self.similarity_matrix = torch.tensor(similarity_matrix, device=self.device, dtype=torch.float32)
            self.n = similarity_matrix.shape[0]
            self.source_embeddings = None
        # Option 2: Compute from embeddings
        elif source_embeddings is not None:
            self.source_embeddings = source_embeddings
            self.n = source_embeddings.shape[0]

            if target_embeddings is None:
                target_embeddings = source_embeddings

            # Compute similarity matrix based on metric
            if metric == 'cosine':
                similarity = cosine_similarity(target_embeddings, source_embeddings)
            elif metric == 'euclidean':
                # Convert Euclidean distance to similarity: smaller distance = higher similarity
                distances = euclidean_distances(target_embeddings, source_embeddings)
                # Use 1/(1+distance) to convert to similarity in [0, 1] range
                similarity = 1.0 / (1.0 + distances)
            else:
                raise ValueError(f"Unknown metric: {metric}. Use 'cosine' or 'euclidean'")

            self.similarity_matrix = torch.tensor(similarity, device=self.device, dtype=torch.float32)
        else:
            raise ValueError("Must provide either source_embeddings or similarity_matrix")

        # Normalize to [0, 1] if any negative values
        min_sim = self.similarity_matrix.min()
        if min_sim < 0:
            self.similarity_matrix = (self.similarity_matrix - min_sim) / (self.similarity_matrix.max() - min_sim)

    def marginal_gain(self, subset, candidate):
        if not subset:
            all_indices = torch.arange(self.n, device=self.device)
            others = all_indices[all_indices != candidate]
            return self.similarity_matrix[candidate, others].sum()

        subset_tensor = torch.tensor(subset, device=self.device)
        all_indices = torch.arange(self.n, device=self.device)
        unselected_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        unselected_mask[subset_tensor] = False
        unselected_mask[candidate] = False
        unselected = all_indices[unselected_mask]

        gain = self.similarity_matrix[candidate, unselected].sum()
        loss = self.similarity_matrix[candidate, subset_tensor].sum()
        marginal_gain = gain - loss

        return marginal_gain

    def lazy_greedy(self, budget):
        subset = []
        gains = torch.zeros(self.n, device=self.device)
        gain_values = []

        for i in range(self.n):
            gains[i] = self.marginal_gain(subset, i)

        for _ in tqdm(range(budget), desc="GraphCut Selection"):
            while True:
                candidate = int(torch.argmax(gains).cpu().numpy())
                exact_gain = self.marginal_gain(subset, candidate)

                if exact_gain >= torch.max(gains) - 1e-6:
                    subset.append(candidate)
                    gain_values.append((candidate, float(exact_gain.cpu().numpy())))
                    gains[candidate] = float('-inf')
                    break
                else:
                    gains[candidate] = exact_gain

        return subset, gain_values
