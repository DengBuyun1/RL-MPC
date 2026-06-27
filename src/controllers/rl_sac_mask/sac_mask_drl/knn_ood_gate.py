from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return row-wise L2-normalized features."""

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norm = np.linalg.norm(arr, ord=2, axis=1, keepdims=True)
    return arr / np.maximum(norm, eps)


def build_state_action_features(
    state_embeddings: np.ndarray,
    actions: np.ndarray,
    action_mean: np.ndarray | float = 0.0,
    action_std: np.ndarray | float = 1.0,
    action_beta: float = 1.0,
) -> np.ndarray:
    """Concatenate normalized state embeddings with standardized RL actions."""

    z = l2_normalize(state_embeddings)
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    mean = np.asarray(action_mean, dtype=np.float32).reshape(1, -1)
    std = np.maximum(np.asarray(action_std, dtype=np.float32).reshape(1, -1), 1e-6)
    a_norm = (a - mean) / std
    return np.concatenate([z, float(action_beta) * a_norm], axis=1).astype(np.float32)


@torch.no_grad()
def extract_actor_history_embedding(agent: Any, history_inputs: torch.Tensor) -> np.ndarray:
    """Extract normalized GRU history embeddings from a trained recurrent actor."""

    device = next(agent.actor.parameters()).device
    history_inputs = history_inputs.to(device)
    features = agent.actor.encoder(history_inputs)
    return l2_normalize(features.detach().cpu().numpy())


def _kth_distances_from_squared(squared_distances: np.ndarray, k: int) -> np.ndarray:
    if squared_distances.shape[1] < k:
        raise ValueError(f"k={k} requires at least {k} reference samples, got {squared_distances.shape[1]}")
    kth_sq = np.partition(squared_distances, k - 1, axis=1)[:, k - 1]
    kth_sq = np.maximum(kth_sq, 0.0)
    return np.sqrt(kth_sq).astype(np.float32)


def kth_neighbor_distances(
    queries: np.ndarray,
    memory: np.ndarray,
    k: int,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute Euclidean distance to the k-th nearest memory vector.

    Inputs are normalized internally. For normalized vectors, squared L2 distance
    is 2 - 2 * cosine_similarity, which avoids materializing feature differences.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    memory_norm = l2_normalize(memory)
    queries_norm = l2_normalize(queries)
    out: list[np.ndarray] = []
    for start in range(0, len(queries_norm), batch_size):
        q = queries_norm[start : start + batch_size]
        squared = 2.0 - 2.0 * np.matmul(q, memory_norm.T)
        out.append(_kth_distances_from_squared(squared, k))
    return np.concatenate(out, axis=0)


def kth_neighbor_indices_and_distances(
    queries: np.ndarray,
    memory: np.ndarray,
    k: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return k nearest-neighbor indices and their distances for each query."""

    if k <= 0:
        raise ValueError("k must be positive")
    memory_norm = l2_normalize(memory)
    queries_norm = l2_normalize(queries)
    all_indices: list[np.ndarray] = []
    all_distances: list[np.ndarray] = []
    for start in range(0, len(queries_norm), batch_size):
        q = queries_norm[start : start + batch_size]
        squared = 2.0 - 2.0 * np.matmul(q, memory_norm.T)
        if squared.shape[1] < k:
            raise ValueError(f"k={k} requires at least {k} reference samples, got {squared.shape[1]}")
        idx = np.argpartition(squared, k - 1, axis=1)[:, :k]
        row = np.arange(len(q))[:, None]
        local_sq = squared[row, idx]
        order = np.argsort(local_sq, axis=1)
        idx = idx[row, order]
        local_sq = local_sq[row, order]
        all_indices.append(idx.astype(np.int64))
        all_distances.append(np.sqrt(np.maximum(local_sq, 0.0)).astype(np.float32))
    return np.concatenate(all_indices, axis=0), np.concatenate(all_distances, axis=0)


def self_kth_neighbor_distances(
    memory: np.ndarray,
    k: int,
    batch_size: int = 2048,
) -> np.ndarray:
    """Compute leave-one-out k-th nearest-neighbor distances within a memory bank."""

    if k <= 0:
        raise ValueError("k must be positive")
    memory_norm = l2_normalize(memory)
    n_samples = len(memory_norm)
    if n_samples <= k:
        raise ValueError(f"leave-one-out calibration with k={k} requires > {k} samples, got {n_samples}")

    out: list[np.ndarray] = []
    all_indices = np.arange(n_samples)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        q = memory_norm[start:stop]
        squared = 2.0 - 2.0 * np.matmul(q, memory_norm.T)
        squared[np.arange(stop - start), all_indices[start:stop]] = np.inf
        out.append(_kth_distances_from_squared(squared, k))
    return np.concatenate(out, axis=0)


def episode_level_kth_neighbor_distances(
    embeddings: np.ndarray,
    episode_ids: np.ndarray,
    k: int,
    batch_size: int = 2048,
) -> np.ndarray:
    """Calibrate distances by holding out one episode at a time."""

    memory = l2_normalize(embeddings)
    episode_ids = np.asarray(episode_ids)
    if len(memory) != len(episode_ids):
        raise ValueError(f"embeddings and episode_ids length mismatch: {len(memory)} vs {len(episode_ids)}")
    out: list[np.ndarray] = []
    for episode_id in np.unique(episode_ids):
        query_mask = episode_ids == episode_id
        ref_mask = ~query_mask
        references = memory[ref_mask]
        queries = memory[query_mask]
        if len(references) < k:
            raise ValueError(
                f"episode-level calibration with k={k} needs at least {k} reference samples; "
                f"episode {episode_id!r} has {len(references)}"
            )
        out.append(kth_neighbor_distances(queries, references, k=k, batch_size=batch_size))
    return np.concatenate(out, axis=0)


@dataclass
class KnnOodGate:
    """KNN-OOD familiarity gate over normalized RL history embeddings."""

    memory: np.ndarray
    threshold: float
    k: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory = l2_normalize(self.memory)
        self.threshold = float(self.threshold)
        self.k = int(self.k)
        if self.k <= 0:
            raise ValueError("k must be positive")
        if len(self.memory) < self.k:
            raise ValueError(f"k={self.k} requires at least {self.k} memory samples, got {len(self.memory)}")

    def score(self, embeddings: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        return kth_neighbor_distances(embeddings, self.memory, self.k, batch_size=batch_size)

    def is_familiar(self, embeddings: np.ndarray, batch_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        distances = self.score(embeddings, batch_size=batch_size)
        return distances <= self.threshold, distances

    def save(
        self,
        path: str | Path,
        calibration_distances: np.ndarray | None = None,
        memory_episode_ids: np.ndarray | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory": self.memory.astype(np.float32),
            "threshold": np.asarray([self.threshold], dtype=np.float32),
            "k": np.asarray([self.k], dtype=np.int32),
            "metadata_json": np.asarray([json.dumps(self.metadata, ensure_ascii=True)]),
        }
        if calibration_distances is not None:
            payload["calibration_distances"] = np.asarray(calibration_distances, dtype=np.float32)
        if memory_episode_ids is not None:
            payload["memory_episode_ids"] = np.asarray(memory_episode_ids)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "KnnOodGate":
        data = np.load(Path(path), allow_pickle=False)
        metadata_json = str(data["metadata_json"][0]) if "metadata_json" in data else "{}"
        return cls(
            memory=data["memory"],
            threshold=float(data["threshold"][0]),
            k=int(data["k"][0]),
            metadata=json.loads(metadata_json),
        )


@dataclass
class ConditionalActionKnnGate:
    """State KNN gate with a local conditional action-support check."""

    memory_state: np.ndarray
    memory_action: np.ndarray
    state_threshold: float
    action_threshold: float
    k_state: int = 50
    action_quantile: float = 0.50
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.memory_state = l2_normalize(self.memory_state)
        self.memory_action = np.asarray(self.memory_action, dtype=np.float32)
        if self.memory_action.ndim == 1:
            self.memory_action = self.memory_action.reshape(-1, 1)
        if len(self.memory_state) != len(self.memory_action):
            raise ValueError("memory_state and memory_action must have the same length")
        self.state_threshold = float(self.state_threshold)
        self.action_threshold = float(self.action_threshold)
        self.k_state = int(self.k_state)
        self.action_quantile = float(self.action_quantile)
        if len(self.memory_state) < self.k_state:
            raise ValueError(f"k_state={self.k_state} requires at least {self.k_state} samples")

    def score(
        self,
        state_embeddings: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray]:
        actions_arr = np.asarray(actions, dtype=np.float32)
        if actions_arr.ndim == 1:
            actions_arr = actions_arr.reshape(-1, 1)
        indices, state_distances = kth_neighbor_indices_and_distances(
            state_embeddings,
            self.memory_state,
            self.k_state,
            batch_size=batch_size,
        )
        state_scores = state_distances[:, -1]
        neighbor_actions = self.memory_action[indices]
        action_diff = np.abs(neighbor_actions - actions_arr[:, None, :])
        if action_diff.shape[-1] == 1:
            action_diff = action_diff[..., 0]
        else:
            action_diff = np.linalg.norm(action_diff, axis=-1)
        action_scores = np.quantile(action_diff, self.action_quantile, axis=1).astype(np.float32)
        return state_scores.astype(np.float32), action_scores

    def is_familiar(
        self,
        state_embeddings: np.ndarray,
        actions: np.ndarray,
        batch_size: int = 4096,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state_scores, action_scores = self.score(state_embeddings, actions, batch_size=batch_size)
        familiar = (state_scores <= self.state_threshold) & (action_scores <= self.action_threshold)
        return familiar, state_scores, action_scores

    def save(
        self,
        path: str | Path,
        calibration_state_distances: np.ndarray | None = None,
        calibration_action_distances: np.ndarray | None = None,
        memory_episode_ids: np.ndarray | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "memory_state": self.memory_state.astype(np.float32),
            "memory_action": self.memory_action.astype(np.float32),
            "state_threshold": np.asarray([self.state_threshold], dtype=np.float32),
            "action_threshold": np.asarray([self.action_threshold], dtype=np.float32),
            "k_state": np.asarray([self.k_state], dtype=np.int32),
            "action_quantile": np.asarray([self.action_quantile], dtype=np.float32),
            "metadata_json": np.asarray([json.dumps(self.metadata, ensure_ascii=True)]),
        }
        if calibration_state_distances is not None:
            payload["calibration_state_distances"] = np.asarray(calibration_state_distances, dtype=np.float32)
        if calibration_action_distances is not None:
            payload["calibration_action_distances"] = np.asarray(calibration_action_distances, dtype=np.float32)
        if memory_episode_ids is not None:
            payload["memory_episode_ids"] = np.asarray(memory_episode_ids)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalActionKnnGate":
        data = np.load(Path(path), allow_pickle=False)
        metadata_json = str(data["metadata_json"][0]) if "metadata_json" in data else "{}"
        return cls(
            memory_state=data["memory_state"],
            memory_action=data["memory_action"],
            state_threshold=float(data["state_threshold"][0]),
            action_threshold=float(data["action_threshold"][0]),
            k_state=int(data["k_state"][0]),
            action_quantile=float(data["action_quantile"][0]),
            metadata=json.loads(metadata_json),
        )


def build_gate(
    embeddings: np.ndarray,
    k: int = 50,
    quantile: float = 0.95,
    batch_size: int = 2048,
    metadata: dict[str, Any] | None = None,
) -> tuple[KnnOodGate, np.ndarray]:
    """Build a KNN-OOD gate and calibrate its threshold from ID embeddings."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    memory = l2_normalize(embeddings)
    calibration_distances = self_kth_neighbor_distances(memory, k=k, batch_size=batch_size)
    threshold = float(np.quantile(calibration_distances, quantile))
    gate = KnnOodGate(memory=memory, threshold=threshold, k=k, metadata=metadata or {})
    return gate, calibration_distances


def build_episode_level_gate(
    embeddings: np.ndarray,
    episode_ids: np.ndarray,
    k: int = 50,
    quantile: float = 0.95,
    batch_size: int = 2048,
    metadata: dict[str, Any] | None = None,
) -> tuple[KnnOodGate, np.ndarray]:
    """Build a KNN-OOD gate with episode-level leave-one-out calibration."""

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    memory = l2_normalize(embeddings)
    calibration_distances = episode_level_kth_neighbor_distances(
        memory,
        episode_ids=episode_ids,
        k=k,
        batch_size=batch_size,
    )
    threshold = float(np.quantile(calibration_distances, quantile))
    gate = KnnOodGate(memory=memory, threshold=threshold, k=k, metadata=metadata or {})
    return gate, calibration_distances


def _conditional_action_scores(
    query_state: np.ndarray,
    query_action: np.ndarray,
    reference_state: np.ndarray,
    reference_action: np.ndarray,
    k_state: int,
    action_quantile: float,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices, state_distances = kth_neighbor_indices_and_distances(
        query_state,
        reference_state,
        k=k_state,
        batch_size=batch_size,
    )
    state_scores = state_distances[:, -1]
    query_action = np.asarray(query_action, dtype=np.float32)
    if query_action.ndim == 1:
        query_action = query_action.reshape(-1, 1)
    reference_action = np.asarray(reference_action, dtype=np.float32)
    if reference_action.ndim == 1:
        reference_action = reference_action.reshape(-1, 1)
    neighbor_actions = reference_action[indices]
    action_diff = np.abs(neighbor_actions - query_action[:, None, :])
    if action_diff.shape[-1] == 1:
        action_diff = action_diff[..., 0]
    else:
        action_diff = np.linalg.norm(action_diff, axis=-1)
    action_scores = np.quantile(action_diff, action_quantile, axis=1).astype(np.float32)
    return state_scores.astype(np.float32), action_scores


def build_episode_level_conditional_action_gate(
    state_embeddings: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    k_state: int = 50,
    state_quantile: float = 0.95,
    action_quantile: float = 0.50,
    action_threshold_quantile: float = 0.95,
    batch_size: int = 2048,
    metadata: dict[str, Any] | None = None,
) -> tuple[ConditionalActionKnnGate, np.ndarray, np.ndarray]:
    """Build a conditional action-support gate with episode-level calibration."""

    if not 0.0 < state_quantile < 1.0:
        raise ValueError("state_quantile must be in (0, 1)")
    if not 0.0 < action_quantile <= 1.0:
        raise ValueError("action_quantile must be in (0, 1]")
    if not 0.0 < action_threshold_quantile < 1.0:
        raise ValueError("action_threshold_quantile must be in (0, 1)")

    states = l2_normalize(state_embeddings)
    actions_arr = np.asarray(actions, dtype=np.float32)
    if actions_arr.ndim == 1:
        actions_arr = actions_arr.reshape(-1, 1)
    episode_ids = np.asarray(episode_ids)
    state_scores_all: list[np.ndarray] = []
    action_scores_all: list[np.ndarray] = []
    for episode_id in np.unique(episode_ids):
        query_mask = episode_ids == episode_id
        ref_mask = ~query_mask
        if not np.any(query_mask):
            continue
        if np.count_nonzero(ref_mask) < k_state:
            continue
        state_scores, action_scores = _conditional_action_scores(
            states[query_mask],
            actions_arr[query_mask],
            states[ref_mask],
            actions_arr[ref_mask],
            k_state=k_state,
            action_quantile=action_quantile,
            batch_size=batch_size,
        )
        state_scores_all.append(state_scores)
        action_scores_all.append(action_scores)
    if not state_scores_all:
        raise ValueError(
            f"episode-level conditional calibration with k_state={k_state} has no valid held-out episodes"
        )
    calibration_state = np.concatenate(state_scores_all, axis=0)
    calibration_action = np.concatenate(action_scores_all, axis=0)
    gate = ConditionalActionKnnGate(
        memory_state=states,
        memory_action=actions_arr,
        state_threshold=float(np.quantile(calibration_state, state_quantile)),
        action_threshold=float(np.quantile(calibration_action, action_threshold_quantile)),
        k_state=k_state,
        action_quantile=action_quantile,
        metadata=metadata or {},
    )
    return gate, calibration_state, calibration_action
