"""
GNN components for wake-aware wind farm control.

Reference: Dai et al., "Learning Combinatorial Optimization Algorithms over Graphs"
           NeurIPS 2017 — Structure2Vec message passing.

Adaptation for wind farms:
  - Nodes  : turbines (state + geographic position + wind context)
  - Edges  : directed wake interactions j→i (j is upstream of i)
  - Edge features : physical wake parameters (distance, direction, deficit)
  - Graph topology changes dynamically with wind direction.

Node features (NODE_DIM = 7):
  [yaw_norm, power_norm, pos_x_norm, pos_y_norm, ws_norm, sin(wd), cos(wd)]

Edge features (EDGE_DIM = 6), for edge j→i:
  [dist_down_norm, dist_cross/D, sin(wd), cos(wd), yaw_j_norm, jensen_deficit]

Message passing (Eq.3 from paper, adapted for directed graph + vector edge features):
  μᵢ^(t+1) = relu( W1·xᵢ  +  W2·Σⱼ∈U(i) μⱼ^(t)  +  W3·Σⱼ∈U(i) relu(W4·eⱼᵢ) )

where U(i) = upstream neighbors of i under current wind direction.
"""

import math
import numpy as np
import torch
import torch.nn as nn


NODE_DIM = 7   # per-turbine node feature dimension
EDGE_DIM = 6   # per-wake-edge feature dimension

_JENSEN_K = 0.075   # wake decay constant
_CT = 0.8           # default thrust coefficient


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

class WakeFarmGraph:
    """
    Builds the dynamic wake-interaction graph from the current farm state.

    Instantiate once per environment (layout is static); call `build()` at
    every timestep to get fresh node features, adjacency, and edge features.

    Returns numpy arrays — callers are responsible for converting to tensors.
    """

    def __init__(
        self,
        layout_x: np.ndarray,
        layout_y: np.ndarray,
        rotor_diameter: float,
        max_yaw: float,
        wind_speed_range: tuple,
        wind_dir_range: tuple,
        rated_power_per_turbine: float,
        jensen_k: float = _JENSEN_K,
        ct: float = _CT,
    ):
        self.lx = np.asarray(layout_x, dtype=np.float64)
        self.ly = np.asarray(layout_y, dtype=np.float64)
        self.n = len(layout_x)
        self.D = rotor_diameter
        self.max_yaw = max_yaw
        self.ws_min, self.ws_max = wind_speed_range
        self.jensen_k = jensen_k
        self.ct = ct
        self.rated_power = rated_power_per_turbine

        # Normalise positions to [0, 1] for node features
        x_range = max(layout_x) - min(layout_x)
        y_range = max(layout_y) - min(layout_y)
        self.norm_x = (self.lx - min(layout_x)) / (x_range if x_range > 0 else 1.0)
        self.norm_y = (self.ly - min(layout_y)) / (y_range if y_range > 0 else 1.0)

        # Diagonal of the farm bounding box — for downwind distance normalisation
        self._max_dist = math.sqrt(x_range ** 2 + y_range ** 2) + 1e-6

    # ------------------------------------------------------------------
    def build(
        self,
        wind_speed: float,
        wind_direction: float,
        yaw_angles: np.ndarray,       # (N,) degrees
        turbine_powers: np.ndarray,   # (N,) watts
    ):
        """
        Construct graph tensors for the current farm state.

        Returns
        -------
        X : np.ndarray  (N, NODE_DIM)   node features
        A : np.ndarray  (N, N)          adjacency;  A[i,j]=1  ⟺  j→i edge exists
        E : np.ndarray  (N, N, EDGE_DIM) edge features (zeros for absent edges)
        """
        N, D, k = self.n, self.D, self.jensen_k

        # ── Wind-aligned coordinate system ─────────────────────────────
        # FLORIS convention: 270° means wind blows in the +x direction.
        # We rotate so that "downwind" is the positive axis.
        wd_rot = math.radians(270.0 - wind_direction)   # rotation angle
        cos_r, sin_r = math.cos(wd_rot), math.sin(wd_rot)

        # For node features: encode wind direction as (sin, cos)
        wd_feat = math.radians(wind_direction)
        sin_wd, cos_wd = math.sin(wd_feat), math.cos(wd_feat)

        ws_norm = (wind_speed - self.ws_min) / max(self.ws_max - self.ws_min, 1e-6)

        # ── Node features ───────────────────────────────────────────────
        X = np.empty((N, NODE_DIM), dtype=np.float32)
        for i in range(N):
            X[i] = [
                yaw_angles[i] / self.max_yaw,          # yaw (normalised)
                turbine_powers[i] / self.rated_power,  # power (normalised)
                float(self.norm_x[i]),                 # geographic x ∈ [0,1]
                float(self.norm_y[i]),                 # geographic y ∈ [0,1]
                ws_norm,                               # wind speed
                sin_wd,                                # wind direction sin
                cos_wd,                                # wind direction cos
            ]

        # ── Edges ───────────────────────────────────────────────────────
        A = np.zeros((N, N), dtype=np.float32)
        E = np.zeros((N, N, EDGE_DIM), dtype=np.float32)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Vector from j to i
                dx = float(self.lx[i] - self.lx[j])
                dy = float(self.ly[i] - self.ly[j])

                # Project to wind-aligned frame
                dist_down  =  dx * cos_r + dy * sin_r   # > 0  ⟺  i is downwind of j
                dist_cross = -dx * sin_r + dy * cos_r   # lateral offset

                if dist_down <= 0:
                    continue   # j is not upstream of i

                # Jensen wake cone: |lateral| < R + k * dist_down
                if abs(dist_cross) >= D / 2 + k * dist_down:
                    continue   # i is outside j's wake

                # ── Edge exists ─────────────────────────────────────────
                A[i, j] = 1.0

                # Jensen wake deficit at i due to j
                yaw_j_rad = math.radians(float(yaw_angles[j]))
                ct_eff = self.ct * math.cos(yaw_j_rad) ** 2
                denom  = D + 2.0 * k * dist_down
                deficit = (1.0 - math.sqrt(max(1.0 - ct_eff, 0.0))) * (D / denom) ** 2

                E[i, j] = [
                    dist_down / self._max_dist,         # normalised downwind distance
                    dist_cross / D,                      # crosswind offset in rotor Ø
                    sin_wd,                              # wind direction (edge context)
                    cos_wd,
                    float(yaw_angles[j]) / self.max_yaw, # upstream yaw (affects deficit)
                    float(deficit),                       # Jensen wake deficit ∈ [0, 1]
                ]

        return X, A, E


# ─────────────────────────────────────────────────────────────────────────────
# Structure2Vec message-passing layer
# ─────────────────────────────────────────────────────────────────────────────

class Structure2VecLayer(nn.Module):
    """
    One iteration of the Structure2Vec update rule (Dai et al. 2017, Eq. 3),
    adapted for directed graphs with multi-dimensional edge features.

    μᵢ^(t+1) = relu( W1·xᵢ  +  W2·Σⱼ∈U(i) μⱼ^(t)  +  W3·Σⱼ∈U(i) relu(W4·eⱼᵢ) )

    All weight matrices are shared across nodes and edges — the network
    is permutation-invariant to turbine ordering and scales to any farm size.
    """

    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        # Matches notation in the paper:
        self.W1 = nn.Linear(node_dim,  embed_dim, bias=False)  # node feature → p
        self.W2 = nn.Linear(embed_dim, embed_dim, bias=False)  # neighbour embed → p
        self.W4 = nn.Linear(edge_dim,  embed_dim, bias=False)  # edge feature → p
        self.W3 = nn.Linear(embed_dim, embed_dim, bias=False)  # aggregated edge → p
        self.bias = nn.Parameter(torch.zeros(embed_dim))

        for lin in (self.W1, self.W2, self.W3, self.W4):
            nn.init.orthogonal_(lin.weight)

    def forward(
        self,
        X:  torch.Tensor,   # (B, N, node_dim)
        mu: torch.Tensor,   # (B, N, embed_dim)  current embeddings
        A:  torch.Tensor,   # (B, N, N)           A[b,i,j]=1 ⟺ j→i
        E:  torch.Tensor,   # (B, N, N, edge_dim)
    ) -> torch.Tensor:      # (B, N, embed_dim)

        B, N, _ = mu.shape

        # Term 1: linear transform of node features
        t1 = self.W1(X)                            # (B, N, p)

        # Term 2: sum of upstream neighbours' embeddings
        #   neighbour_agg[b, i, :] = Σⱼ  A[b,i,j] · μⱼ
        neighbour_agg = torch.bmm(A, mu)            # (B, N, p)
        t2 = self.W2(neighbour_agg)                 # (B, N, p)

        # Term 3: aggregated (transformed) edge features
        #   E[b,i,j] → relu(W4·E[b,i,j]) → sum over j where A[b,i,j]=1
        E_flat = E.view(B, N * N, -1)              # (B, N², edge_dim)
        ef = torch.relu(self.W4(E_flat))            # (B, N², p)
        ef = ef.view(B, N, N, -1)                   # (B, N, N, p)
        edge_agg = (A.unsqueeze(-1) * ef).sum(dim=2)  # (B, N, p)
        t3 = self.W3(edge_agg)                      # (B, N, p)

        return torch.relu(t1 + t2 + t3 + self.bias)  # (B, N, p)


# ─────────────────────────────────────────────────────────────────────────────
# GNN Encoder  (T layers of Structure2Vec)
# ─────────────────────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """
    T-layer Structure2Vec encoder.

    Maps (X, A, E) → per-node embeddings of dimension `embed_dim`.

    Handles both batched (B, N, …) and single-sample (N, …) inputs.
    """

    def __init__(
        self,
        node_dim:  int = NODE_DIM,
        edge_dim:  int = EDGE_DIM,
        embed_dim: int = 64,
        n_layers:  int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Project raw node features into the embedding space (μ⁰)
        self.init_embed = nn.Linear(node_dim, embed_dim)
        nn.init.orthogonal_(self.init_embed.weight)

        # T independent Structure2Vec layers (separate parameters → more expressive)
        self.layers = nn.ModuleList([
            Structure2VecLayer(node_dim, edge_dim, embed_dim)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        X: torch.Tensor,   # (B, N, node_dim)  or  (N, node_dim)
        A: torch.Tensor,   # (B, N, N)          or  (N, N)
        E: torch.Tensor,   # (B, N, N, edge_dim) or (N, N, edge_dim)
    ) -> torch.Tensor:     # (B, N, embed_dim)   or (N, embed_dim)

        single = X.dim() == 2          # True if no batch dimension
        if single:
            X, A, E = X.unsqueeze(0), A.unsqueeze(0), E.unsqueeze(0)

        mu = torch.relu(self.init_embed(X))   # (B, N, p)  initial embeddings

        for layer in self.layers:
            mu = layer(X, mu, A, E)           # (B, N, p)

        return mu.squeeze(0) if single else mu
