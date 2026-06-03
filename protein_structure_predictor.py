import os
import json
import time
import warnings
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

try:
    from Bio import PDB
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_OK = True
except ImportError:
    BIOPYTHON_OK = False
    print("[WARN] Biopython not found. Install with: pip install biopython")



# GLOBAL CONFIGURATION
CONFIG = {
    # Data
    "data_dir"        : "pdb_data",
    "output_dir"      : "outputs",
    "checkpoint_dir"  : "checkpoints",
    "max_proteins"    : 600,       # How many PDB entries to attempt to download
    "min_seq_len"     : 30,        # Shortest protein to keep
    "max_seq_len"     : 128,       # Padded/truncated length (N)
    "max_resolution"  : 2.5,       # X-ray resolution filter (Å)
    # Model
    "embed_dim"       : 128,       # Residue embedding dimension
    "cnn_channels"    : [128, 256, 256],  # 1D CNN output channels per block
    "cnn_kernel"      : 5,         # Convolutional kernel width
    "nhead"           : 8,         # Transformer multi-head attention heads
    "num_tf_layers"   : 4,         # Transformer encoder depth
    "mlp_hidden"      : 256,       # MLP regression head hidden size
    "dropout"         : 0.10,
    # Training
    "batch_size"      : 16,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-4,
    "epochs"          : 60,
    "patience"        : 12,        # Early stopping patience
    "grad_clip"       : 1.0,
    # Loss
    "contact_cutoff"  : 8.0,       # Å – contacts get extra loss weight
    "contact_weight"  : 2.0,       # Multiplier on contact pairs
    # Hardware
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
    "seed"            : 42,
}



# AMINO ACID TOKENISATION

# 20 canonical amino acids + PAD (0) + UNK (21)
AA_VOCAB: dict = {
    "A": 1,  "C": 2,  "D": 3,  "E": 4,  "F": 5,
    "G": 6,  "H": 7,  "I": 8,  "K": 9,  "L": 10,
    "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15,
    "S": 16, "T": 17, "V": 18, "W": 19, "Y": 20,
    "<PAD>": 0, "<UNK>": 21,
}
VOCAB_SIZE = len(AA_VOCAB)  # 22

# Three-letter → one-letter map (used during PDB parsing)
_THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}



# DATA: DOWNLOAD & PARSE

def fetch_pdb_ids(max_results: int, min_len: int, max_len: int,
                  max_resolution: float) -> List[str]:
    """
    Query the RCSB PDB REST API for single-chain protein structures that fall
    within our length and resolution bounds.  Falls back to a curated list if
    the API is unavailable.
    """
    query_payload = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": max_resolution,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "greater_or_equal",
                        "value": min_len,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "less_or_equal",
                        "value": max_len,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_results},
            "sort": [{"sort_by": "score", "direction": "desc"}],
        },
    }

    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    try:
        r = requests.post(url, json=query_payload, timeout=30)
        r.raise_for_status()
        ids = [hit["identifier"] for hit in r.json().get("result_set", [])]
        print(f"[DATA]  RCSB query returned {len(ids)} entries.")
        return ids
    except Exception as exc:
        print(f"[DATA]  RCSB query failed ({exc}). Using fallback PDB ID list.")
        # A hand-curated set of short, well-characterised proteins
        return [
            "1CRN", "1UBQ", "1VII", "1L2Y", "2GB1",
            "1FSD", "1BDD", "2HBA", "1SHF", "2LZM",
            "1MBN", "3LZT", "2CI2", "1PGB", "1TIM",
            "1ENH", "2HHB", "1HRC", "1BPI", "1LMB",
            "1AHO", "1E0L", "1LFC", "2PTL", "1BX7",
            "1WIT", "3GB1", "1IGD", "2JOF", "1QYS",
        ]


def download_pdb_file(pdb_id: str, data_dir: str) -> Optional[str]:
    """Download a single PDB file; skip if already on disk."""
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"{pdb_id.lower()}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(out_path, "w") as fh:
            fh.write(r.text)
        return out_path
    except Exception:
        return None


def parse_ca_from_pdb(filepath: str, max_len: int) -> Optional[Tuple[str, np.ndarray]]:
    """
    Extract the first valid polypeptide chain from a PDB file.

    Returns
    
    (sequence, coords) where
        sequence : str  – single-letter amino acid sequence (length L ≤ max_len)
        coords   : (L, 3) float32 array of Cα positions in Ångströms
    """
    if not BIOPYTHON_OK:
        return None
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("prot", filepath)
    except Exception:
        return None

    for chain in structure[0]:         # first MODEL, iterate chains
        seq_chars, ca_coords = [], []
        for residue in chain.get_residues():
            if not is_aa(residue, standard=True):
                continue
            if "CA" not in residue:
                continue
            aa1 = _THREE_TO_ONE.get(residue.get_resname().strip())
            if aa1 is None:
                continue
            seq_chars.append(aa1)
            ca_coords.append(residue["CA"].get_vector().get_array())

        L = len(seq_chars)
        if L < 10:
            continue
        if L > max_len:                # truncate to max_len
            seq_chars = seq_chars[:max_len]
            ca_coords = ca_coords[:max_len]

        return "".join(seq_chars), np.array(ca_coords, dtype=np.float32)

    return None


def build_dataset(cfg: dict) -> List[Tuple[str, np.ndarray]]:
    """
    Build a list of (sequence, Cα-coords) pairs.
    Results are cached to disk so subsequent runs are instant.
    """
    cache_path = os.path.join(cfg["data_dir"], "dataset_cache.json")

    if os.path.exists(cache_path):
        print("[DATA]  Loading cached dataset from disk …")
        with open(cache_path) as fh:
            raw = json.load(fh)
        data = [(d["seq"], np.array(d["coords"], dtype=np.float32)) for d in raw]
        print(f"[DATA]  Loaded {len(data)} proteins from cache.")
        return data

    pdb_ids = fetch_pdb_ids(
        cfg["max_proteins"], cfg["min_seq_len"],
        cfg["max_seq_len"], cfg["max_resolution"],
    )

    dataset, failed = [], 0
    iterator = tqdm(pdb_ids, desc="Downloading PDB") if USE_TQDM else pdb_ids

    for i, pid in enumerate(iterator):
        path = download_pdb_file(pid, cfg["data_dir"])
        if path is None:
            failed += 1
            continue
        result = parse_ca_from_pdb(path, cfg["max_seq_len"])
        if result is None:
            failed += 1
            continue
        seq, coords = result
        if len(seq) < cfg["min_seq_len"]:
            continue
        dataset.append((seq, coords))
        if not USE_TQDM and (i + 1) % 50 == 0:
            print(f"[DATA]  {i+1}/{len(pdb_ids)}  collected={len(dataset)}  failed={failed}")
        time.sleep(0.03)               # polite rate-limit

    print(f"[DATA]  Final dataset: {len(dataset)} proteins  (failed={failed})")

    os.makedirs(cfg["data_dir"], exist_ok=True)
    cache_data = [{"seq": s, "coords": c.tolist()} for s, c in dataset]
    with open(cache_path, "w") as fh:
        json.dump(cache_data, fh)

    return dataset



# PYTORCH DATASET

class ProteinDataset(Dataset):
    """
    Wraps (sequence, Cα-coords) pairs into padded tensors.

    Each sample returns:
        tokens  : (N,)   LongTensor of amino-acid indices (0-padded)
        mask    : (N,)   FloatTensor, 1 = real residue, 0 = padding
        coords  : (N, 3) FloatTensor, Cα coordinates (0-padded, centred)
        length  : int    actual sequence length L
    """

    def __init__(self, data: List[Tuple[str, np.ndarray]], max_len: int):
        self.data = data
        self.N = max_len

    def __len__(self) -> int:
        return len(self.data)

    def _tokenise(self, seq: str) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in seq]
        L = len(ids)
        pad = self.N - L
        ids += [AA_VOCAB["<PAD>"]] * pad
        mask = [1.0] * L + [0.0] * pad
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def _pad_coords(self, coords: np.ndarray, L: int) -> np.ndarray:
        """Centre (remove mean) and zero-pad to (N, 3)."""
        centred = coords - coords.mean(axis=0, keepdims=True)
        pad_size = self.N - L
        if pad_size > 0:
            centred = np.concatenate(
                [centred, np.zeros((pad_size, 3), dtype=np.float32)], axis=0
            )
        return centred

    def __getitem__(self, idx: int):
        seq, coords = self.data[idx]
        L = len(seq)
        tokens, mask = self._tokenise(seq)
        coords_padded = torch.tensor(
            self._pad_coords(coords, L), dtype=torch.float32
        )
        return tokens, mask, coords_padded, L


def collate_fn(batch):
    tokens, masks, coords, lengths = zip(*batch)
    return (
        torch.stack(tokens),                        # (B, N)
        torch.stack(masks),                         # (B, N)
        torch.stack(coords),                        # (B, N, 3)
        torch.tensor(lengths, dtype=torch.long),    # (B,)
    )


# MODEL ARCHITECTURE

class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).
    Adds a fixed position signal to the residue embeddings.
    """

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10_000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class CaStructurePredictor(nn.Module):
    """
    End-to-end model: amino-acid tokens → 3D Cα coordinates.

    Pipeline
    
    1. Embedding layer         – maps each token to a D-dim vector
    2. Sinusoidal positional   – injects sequential position information
    3. 1D CNN blocks           – extract local secondary-structure motifs
       (residual add back to embedding stream)
    4. Transformer Encoder     – captures global (long-range) pairwise
       interactions via multi-head self-attention
    5. MLP regression head     – projects each position's representation
       to (x, y, z) Cα coordinates
    """

    def __init__(self, cfg: dict):
        super().__init__()
        D   = cfg["embed_dim"]
        N   = cfg["max_seq_len"]
        do  = cfg["dropout"]
        cnn_ch   = cfg["cnn_channels"]
        kern     = cfg["cnn_kernel"]
        nhead    = cfg["nhead"]
        n_layers = cfg["num_tf_layers"]
        mlp_h    = cfg["mlp_hidden"]

        # ── 1. Embedding ────────────────────────────────────────────────────
        self.embedding  = nn.Embedding(VOCAB_SIZE, D, padding_idx=0)
        self.pos_enc    = SinusoidalPositionalEncoding(D, N, do)
        self.embed_norm = nn.LayerNorm(D)

        # ── 2. 1D CNN local feature extractor ───────────────────────────────
        # Input : (B, D, L)  →  Output : (B, cnn_ch[-1], L)
        cnn_layers = []
        in_ch = D
        for out_ch in cnn_ch:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kern, padding=kern // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(do),
            ]
            in_ch = out_ch
        self.cnn      = nn.Sequential(*cnn_layers)
        self.cnn_proj = nn.Linear(cnn_ch[-1], D)   # back to D for residual add

        # ── 3. Transformer Encoder ───────────────────────────────────────────
        tf_layer = nn.TransformerEncoderLayer(
            d_model         = D,
            nhead           = nhead,
            dim_feedforward = D * 4,
            dropout         = do,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,    # Pre-LN: more stable gradient flow
        )
        self.transformer = nn.TransformerEncoder(tf_layer, num_layers=n_layers)

        # ── 4. MLP regression head: (B, N, D) → (B, N, 3) 
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(D),
            nn.Linear(D, mlp_h),
            nn.GELU(),
            nn.Dropout(do),
            nn.Linear(mlp_h, mlp_h // 2),
            nn.GELU(),
            nn.Linear(mlp_h // 2, 3),
        )

    def forward(
        self,
        tokens: torch.Tensor,       # (B, N)  LongTensor
        mask:   torch.Tensor,       # (B, N)  FloatTensor  1=real 0=pad
    ) -> torch.Tensor:              # (B, N, 3)
        # Boolean padding mask for Transformer: True ↔ ignore position
        key_pad_mask = mask == 0                        # (B, N)

        # Embedding + positional encoding
        x = self.embed_norm(self.pos_enc(self.embedding(tokens)))  # (B, N, D)

        # CNN (needs channel-first layout)
        x_cnn = self.cnn(x.permute(0, 2, 1))           # (B, C, N)
        x_cnn = self.cnn_proj(x_cnn.permute(0, 2, 1))  # (B, N, D)
        x = x + x_cnn                                  # residual

        # Transformer (global attention across all residues)
        x = self.transformer(x, src_key_padding_mask=key_pad_mask)  # (B, N, D)

        # Regression head → coordinates
        coords = self.mlp_head(x)                       # (B, N, 3)

        # Zero out padded positions (no signal in padding)
        coords = coords * mask.unsqueeze(-1)

        return coords


# LOSS FUNCTION  –  Pairwise Distance Matrix Loss

def batch_pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    """
    Efficient batched pairwise Euclidean distance matrix.

    Args:
        coords : (B, N, 3)
    Returns:
        dists  : (B, N, N)  symmetric, zero diagonal
    """
    sq_norms  = (coords ** 2).sum(dim=-1, keepdim=True)            # (B, N, 1)
    dots      = torch.bmm(coords, coords.transpose(1, 2))          # (B, N, N)
    sq_dists  = (sq_norms + sq_norms.transpose(1, 2) - 2.0 * dots).clamp(min=0.0)
    return torch.sqrt(sq_dists + 1e-8)


def distance_matrix_loss(
    pred_coords   : torch.Tensor,       # (B, N, 3)
    true_coords   : torch.Tensor,       # (B, N, 3)
    mask          : torch.Tensor,       # (B, N)  1=real
    contact_cutoff: float = 8.0,
    contact_weight: float = 2.0,
) -> torch.Tensor:
    """
    Rotation- and translation-invariant structure loss.

    We compare predicted and true *pairwise distance matrices* rather than
    raw coordinates.  This handles the fundamental degeneracy that a protein
    looks the same from any direction.

    An additional contact-map weighting upweights pairs closer than
    `contact_cutoff` Å, encouraging the model to correctly place interacting
    residue pairs that define tertiary topology.
    """
    # Pair-level mask: 1 only if BOTH residues are real (not padding)
    pair_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(-2))  # (B, N, N)

    pred_dm = batch_pairwise_distances(pred_coords)                # (B, N, N)
    true_dm = batch_pairwise_distances(true_coords)                # (B, N, N)

    sq_err = (pred_dm - true_dm) ** 2                              # (B, N, N)

    # Upweight close pairs (contacts define fold topology)
    contact_mask = (true_dm < contact_cutoff).float()
    weight_map   = 1.0 + (contact_weight - 1.0) * contact_mask    # (B, N, N)

    loss = (sq_err * weight_map * pair_mask).sum() / (pair_mask.sum() + 1e-8)
    return loss


# EVALUATION METRICS

def drmsd(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Distance-RMSD: rotation/translation-invariant quality metric.

    RMSD over all unique pairwise inter-residue distances.
    Lower is better; units = Ångströms.
    """
    idx_i, idx_j = np.triu_indices(len(pred), k=1)
    p_d = np.linalg.norm(pred[idx_i] - pred[idx_j], axis=-1)
    t_d = np.linalg.norm(true[idx_i] - true[idx_j], axis=-1)
    return float(np.sqrt(np.mean((p_d - t_d) ** 2)))


def contact_precision(pred: np.ndarray, true: np.ndarray,
                      threshold: float = 8.0) -> float:
    """
    Fraction of predicted contacts (< threshold Å) that are true contacts.
    """
    idx_i, idx_j = np.triu_indices(len(pred), k=1)
    p_d = np.linalg.norm(pred[idx_i] - pred[idx_j], axis=-1)
    t_d = np.linalg.norm(true[idx_i] - true[idx_j], axis=-1)
    pred_contacts = p_d < threshold
    if pred_contacts.sum() == 0:
        return 0.0
    true_contacts = t_d < threshold
    return float((pred_contacts & true_contacts).sum() / pred_contacts.sum())


# TRAINING LOOP

def run_epoch(
    model, loader, optimizer, device, cfg, training: bool
) -> Tuple[float, float, float]:
    """One full epoch.  Returns (mean_loss, mean_drmsd, mean_contact_prec)."""
    model.train(training)
    total_loss, drmsds, cprecs = 0.0, [], []

    with torch.set_grad_enabled(training):
        for tokens, mask, coords, lengths in loader:
            tokens  = tokens.to(device)
            mask    = mask.to(device)
            coords  = coords.to(device)

            pred = model(tokens, mask)
            loss = distance_matrix_loss(
                pred, coords, mask,
                cfg["contact_cutoff"], cfg["contact_weight"],
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                optimizer.step()

            total_loss += loss.item()

            # Per-sample evaluation metrics (CPU)
            pred_np    = pred.detach().cpu().numpy()
            true_np    = coords.cpu().numpy()
            lengths_np = lengths.numpy()
            for b, L in enumerate(lengths_np):
                drmsds.append(drmsd(pred_np[b, :L], true_np[b, :L]))
                cprecs.append(contact_precision(
                    pred_np[b, :L], true_np[b, :L], cfg["contact_cutoff"]
                ))

    return (
        total_loss / len(loader),
        float(np.mean(drmsds)),
        float(np.mean(cprecs)),
    )


# VISUALISATION

def plot_training_curves(
    train_losses, val_losses, val_drmsds, val_cprec, save_dir: str
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(train_losses) + 1)

    axes[0].plot(ep, train_losses, label="Train", color="steelblue")
    axes[0].plot(ep, val_losses,   label="Val",   color="coral")
    axes[0].set_title("Distance-Matrix Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(ep, val_drmsds, color="green")
    axes[1].set_title("Validation dRMSD")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("dRMSD (Å)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ep, val_cprec, color="purple")
    axes[2].set_title("Contact Precision (val)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Precision")
    axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("[PLOT]  Saved training_curves.png")


def plot_3d_structure(
    pred_coords: np.ndarray, true_coords: np.ndarray,
    label: str, save_dir: str
):
    """Side-by-side 3D scatter of true vs predicted Cα backbone."""
    fig = plt.figure(figsize=(12, 5))
    for col_idx, (coords, title, color) in enumerate([
        (true_coords, f"True – {label}",      "steelblue"),
        (pred_coords, f"Predicted – {label}", "coral"),
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        ax.plot(
            coords[:, 0], coords[:, 1], coords[:, 2],
            "-o", color=color, markersize=3, linewidth=1.5, alpha=0.85,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")

    plt.suptitle(
        f"Cα backbone comparison  |  dRMSD = {drmsd(pred_coords, true_coords):.2f} Å",
        fontsize=12,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"structure_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT]  Saved structure_{label}.png")


def plot_distance_matrices(
    pred_coords: np.ndarray, true_coords: np.ndarray,
    label: str, save_dir: str
):
    """
    Show true distance matrix, predicted distance matrix, and absolute error –
    a standard diagnostic for structure-prediction models.
    """
    from scipy.spatial.distance import cdist
    pred_dm = cdist(pred_coords, pred_coords)
    true_dm = cdist(true_coords, true_coords)
    err_dm  = np.abs(pred_dm - true_dm)
    vmax = true_dm.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles  = ["True Distance Matrix", "Predicted Distance Matrix", "|Pred − True|"]
    dms     = [true_dm, pred_dm, err_dm]
    cmaps   = ["viridis", "viridis", "hot"]
    vmaxes  = [vmax, vmax, None]

    for ax, dm, ttl, cmap, vm in zip(axes, dms, titles, cmaps, vmaxes):
        kw = dict(cmap=cmap, vmin=0)
        if vm is not None:
            kw["vmax"] = vm
        im = ax.imshow(dm, **kw)
        ax.set_title(ttl); ax.set_xlabel("Residue"); ax.set_ylabel("Residue")
        plt.colorbar(im, ax=ax, label="Distance (Å)")

    plt.suptitle(
        f"{label}  |  dRMSD = {drmsd(pred_coords, true_coords):.2f} Å  "
        f"|  Contact prec = {contact_precision(pred_coords, true_coords):.2%}",
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"distance_matrix_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT]  Saved distance_matrix_{label}.png")


def plot_sequence_attention(
    model: nn.Module, seq: str, max_len: int, device: str, label: str,
    save_dir: str,
):
    """
    Visualise the attention weights of the first Transformer layer
    to see which residue pairs the model attends to most.
    """
    tokens = torch.tensor(
        [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in seq]
        + [0] * (max_len - len(seq)),
        dtype=torch.long,
    ).unsqueeze(0).to(device)
    mask = torch.tensor(
        [1.0] * len(seq) + [0.0] * (max_len - len(seq)),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    # Register a hook to capture attention weights
    attn_weights = {}

    def _hook(module, inp, out):
        # TransformerEncoderLayer stores attn weights when need_weights=True
        # We re-run the attention sub-layer manually
        pass

    # Simpler: run a forward pass on a minimal sub-model
    model.eval()
    with torch.no_grad():
        x = model.embed_norm(model.pos_enc(model.embedding(tokens)))
        x_cnn = model.cnn(x.permute(0, 2, 1))
        x_cnn = model.cnn_proj(x_cnn.permute(0, 2, 1))
        x = x + x_cnn
        # Manually run first TF layer with need_weights=True
        layer = model.transformer.layers[0]
        # Pre-norm
        x_n = layer.norm1(x)
        attn_out, attn_w = layer.self_attn(
            x_n, x_n, x_n,
            key_padding_mask=(mask == 0),
            need_weights=True,
            average_attn_weights=True,
        )
        # attn_w : (B, N, N)
        attn = attn_w[0, :len(seq), :len(seq)].cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap="Blues", aspect="auto")
    ax.set_title(f"Self-Attention (layer 1)  –  {label}")
    ax.set_xlabel("Key residue"); ax.set_ylabel("Query residue")
    if len(seq) <= 30:
        ax.set_xticks(range(len(seq))); ax.set_xticklabels(list(seq), fontsize=6)
        ax.set_yticks(range(len(seq))); ax.set_yticklabels(list(seq), fontsize=6)
    plt.colorbar(im, ax=ax, label="Attention weight")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"attention_{label}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT]  Saved attention_{label}.png")

# PREDICTION UTILITY

@torch.no_grad()
def predict(
    model: nn.Module,
    sequence: str,
    max_len: int,
    device: str,
) -> np.ndarray:
    """
    Predict Cα coordinates for a single amino-acid sequence.

    Parameters
    
    model    : trained CaStructurePredictor
    sequence : str  (one-letter amino-acid codes, e.g. "QKSALVAKVS")
    max_len  : int  (must match the max_len used during training)
    device   : str  "cpu" or "cuda"

    Returns
    
    coords : (L, 3) numpy array of predicted Cα positions in Å
    """
    model.eval()
    L = min(len(sequence), max_len)
    seq = sequence[:L]

    tokens = torch.tensor(
        [AA_VOCAB.get(aa, AA_VOCAB["<UNK>"]) for aa in seq]
        + [0] * (max_len - L),
        dtype=torch.long,
    ).unsqueeze(0).to(device)

    mask = torch.tensor(
        [1.0] * L + [0.0] * (max_len - L),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    pred = model(tokens, mask)                  # (1, N, 3)
    return pred[0, :L].cpu().numpy()



# MAIN

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    device = CONFIG["device"]
    N      = CONFIG["max_seq_len"]
    print(f"[INFO]  Device : {device}")
    print(f"[INFO]  Max sequence length : {N}")

    #1. Data 
    print("\n" + "="*60)
    print(" STEP 1 – Data")
    print("="*60)
    dataset = build_dataset(CONFIG)
    if len(dataset) < 20:
        raise RuntimeError(
            f"Only {len(dataset)} proteins available – need at least 20.  "
            "Check your internet connection or reduce min_seq_len / max_resolution."
        )

    rng = np.random.default_rng(CONFIG["seed"])
    idx = rng.permutation(len(dataset))
    n_train = int(0.80 * len(dataset))
    n_val   = int(0.10 * len(dataset))

    train_data = [dataset[i] for i in idx[:n_train]]
    val_data   = [dataset[i] for i in idx[n_train : n_train + n_val]]
    test_data  = [dataset[i] for i in idx[n_train + n_val :]]
    print(f"[DATA]  Split  train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    train_loader = DataLoader(
        ProteinDataset(train_data, N), batch_size=CONFIG["batch_size"],
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        ProteinDataset(val_data, N), batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    test_loader = DataLoader(
        ProteinDataset(test_data, N), batch_size=CONFIG["batch_size"],
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    # 2. Model
    print("\n" + "="*60)
    print(" STEP 2 – Model")
    print("="*60)
    model     = CaStructurePredictor(CONFIG).to(device)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Trainable parameters : {n_params:,}")
    print(model)

    optimizer = optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["epochs"], eta_min=1e-6
    )

    #3. Training
    print("\n" + "="*60)
    print(" STEP 3 – Training")
    print("="*60)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    train_losses, val_losses, val_drmsds, val_cprecs = [], [], [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        tr_loss, tr_drmsd, tr_cp = run_epoch(
            model, train_loader, optimizer, device, CONFIG, training=True
        )
        vl_loss, vl_drmsd, vl_cp = run_epoch(
            model, val_loader, optimizer, device, CONFIG, training=False
        )
        scheduler.step()

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_drmsds.append(vl_drmsd)
        val_cprecs.append(vl_cp)

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:3d}/{CONFIG['epochs']}  "
            f"| train loss {tr_loss:7.4f}  dRMSD {tr_drmsd:6.2f} Å  "
            f"| val loss {vl_loss:7.4f}  dRMSD {vl_drmsd:6.2f} Å  "
            f"cp {vl_cp:.2%}  | lr {lr_now:.1e}"
        )

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch"    : epoch,
                    "model"    : model.state_dict(),
                    "optim"    : optimizer.state_dict(),
                    "val_loss" : vl_loss,
                    "val_drmsd": vl_drmsd,
                    "config"   : CONFIG,
                },
                os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"[TRAIN] Early stopping at epoch {epoch}.")
                break

    # 4. Test-set evaluation
    print("\n" + "="*60)
    print(" STEP 4 – Test Evaluation")
    print("="*60)
    ckpt = torch.load(
        os.path.join(CONFIG["checkpoint_dir"], "best_model.pt"),
        map_location=device,
    )
    model.load_state_dict(ckpt["model"])
    te_loss, te_drmsd, te_cp = run_epoch(
        model, test_loader, None, device, CONFIG, training=False
    )
    print(f"[RESULT]  Test loss  = {te_loss:.4f}")
    print(f"[RESULT]  Test dRMSD = {te_drmsd:.2f} Å")
    print(f"[RESULT]  Contact precision = {te_cp:.2%}")

    #5. Plots
    print("\n" + "="*60)
    print(" STEP 5 – Visualisation")
    print("="*60)
    out_dir = CONFIG["output_dir"]

    plot_training_curves(
        train_losses, val_losses, val_drmsds, val_cprecs, out_dir
    )

    # Visualise 3 test proteins
    model.eval()
    for k, (seq, true_coords) in enumerate(test_data[:3]):
        pred_coords = predict(model, seq, N, device)
        label       = f"test_{k+1}_L{len(seq)}"
        plot_3d_structure(pred_coords, true_coords, label, out_dir)
        plot_distance_matrices(pred_coords, true_coords, label, out_dir)
        plot_sequence_attention(model, seq, N, device, label, out_dir)

    #6. Demo inference 
    print("\n" + "="*60)
    print(" STEP 6 – Demo Inference (from raw sequence)")
    print("="*60)
    demo_seq = "QKSALVAKVSDGQSTLSITVENKATITFTNITEVSKHFEQLSEGKAQMLEELKQ"
    demo_coords = predict(model, demo_seq, N, device)
    print(f"[DEMO]  Sequence : {demo_seq}")
    print(f"[DEMO]  Predicted Cα shape : {demo_coords.shape}  (L={len(demo_seq)} residues)")
    print(f"[DEMO]  Coordinate range   : [{demo_coords.min():.1f}, {demo_coords.max():.1f}] Å")

    # Save demo structure plot
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")
    ax.plot(
        demo_coords[:, 0], demo_coords[:, 1], demo_coords[:, 2],
        "-o", color="steelblue", markersize=4, linewidth=1.8,
    )
    ax.set_title("Demo – Predicted Cα backbone", fontsize=11)
    ax.set_xlabel("X (Å)"); ax.set_ylabel("Y (Å)"); ax.set_zlabel("Z (Å)")
    plt.tight_layout()
    demo_path = os.path.join(out_dir, "demo_prediction.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(demo_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[DEMO]  Saved demo_prediction.png")

    print("\n" + "="*60)
    print(" DONE")
    print("="*60)
    print(f"  Best checkpoint : {CONFIG['checkpoint_dir']}/best_model.pt")
    print(f"  Plots           : {CONFIG['output_dir']}/")
    print(f"  Final test dRMSD: {te_drmsd:.2f} Å  (contact precision: {te_cp:.2%})")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
