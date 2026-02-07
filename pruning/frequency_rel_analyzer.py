import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model_utils import ModelUtils
import numpy as np
import scipy.fft as sfft
from typing import Dict, Tuple, List, Sequence, Optional
class FrequencyRelevanceAnalyzer:
    def __init__(self, config):
        self.config = config
    def build_frequency_relevance_net(
        self,
        hidden_units: Sequence[int] = (64, 32),
        input_dim: int = 3,
        architecture: Optional[str] = None,
        num_classes: int = 3,
    ) -> nn.Module:
        """
        Build the FRN backbone.
        Args:
            hidden_units: base hidden sizes.
            input_dim: number of input features.
            architecture: 'residual' for residual MLP, 'dense' for vanilla stack.
        """
        arch = (architecture or getattr(self.config, "frn_architecture", "residual")).lower()
        if arch not in {"residual", "dense"}:
            arch = "residual"
        units_seq = tuple(int(u) for u in hidden_units if int(u) > 0) or (64, 32)
        use_batchnorm = bool(getattr(self.config, "frn_use_batchnorm", True))
        dropout_cfg = getattr(self.config, "frn_dropout_rate", 0.05)
        if isinstance(dropout_cfg, (list, tuple)):
            dropout_rates = [float(rate) for rate in dropout_cfg] or [0.05]
        else:
            dropout_rates = [float(dropout_cfg)]
        if len(dropout_rates) < len(units_seq):
            dropout_rates.extend([dropout_rates[-1]] * (len(units_seq) - len(dropout_rates)))
        else:
            dropout_rates = dropout_rates[: len(units_seq)]
        if len(units_seq) > 0:
            final_rate = dropout_rates[-1]
            dropout_rates = [0.0] * (len(units_seq) - 1) + [final_rate]

        class FRN(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                prev_dim = input_dim
                for idx, units in enumerate(units_seq, start=1):
                    if arch == "dense":
                        self.layers.append(nn.Linear(prev_dim, units))
                        if use_batchnorm:
                            self.layers.append(nn.BatchNorm1d(units))
                        self.layers.append(nn.ReLU())
                        drop_rate = dropout_rates[idx - 1]
                        if drop_rate > 0.0:
                            self.layers.append(nn.Dropout(drop_rate))
                        prev_dim = units
                    else:  # residual
                        proj = nn.Linear(prev_dim, units) if prev_dim != units else None
                        dense1 = nn.Linear(prev_dim, units)
                        bn1 = nn.BatchNorm1d(units) if use_batchnorm else None
                        relu1 = nn.ReLU()
                        drop1 = nn.Dropout(dropout_rates[idx - 1]) if dropout_rates[idx - 1] > 0.0 else None
                        dense2 = nn.Linear(units, units)
                        bn2 = nn.BatchNorm1d(units) if use_batchnorm else None
                        drop2 = nn.Dropout(dropout_rates[idx - 1]) if dropout_rates[idx - 1] > 0.0 else None
                        self.layers.extend([proj, dense1, bn1, relu1, drop1, dense2, bn2, drop2])
                        prev_dim = units
                self.fc_out = nn.Linear(prev_dim, num_classes)

            def forward(self, x):
                for layer in self.layers:
                    if layer is None:
                        continue
                    if isinstance(layer, nn.Linear) and 'proj' in layer._get_name():  # simple check
                        x = layer(x)
                    elif isinstance(layer, nn.Dropout) or isinstance(layer, nn.ReLU) or isinstance(layer, nn.BatchNorm1d):
                        x = layer(x)
                    else:
                        residual = x
                        x = layer(x)
                return F.softmax(self.fc_out(x), dim=-1)

        return FRN()
    # ---------------------------- utilities ---------------------------- #
    def dct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT-II with orthonormal scaling over the spatial axes."""
        # x: [H, W, Cin, Cout]
        # DCT along H (dim=0)
        x_h = x.permute(1, 2, 3, 0).numpy()
        y_h = sfft.dct(x_h, axis=-1, type=2, norm='ortho')
        y_h = torch.from_numpy(y_h).permute(3, 0, 1, 2)
        # DCT along W (dim=1)
        x_w = y_h.permute(0, 2, 3, 1).numpy()
        y_w = sfft.dct(x_w, axis=-1, type=2, norm='ortho')
        return torch.from_numpy(y_w).permute(0, 3, 1, 2)
    def idct2_ortho(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse 2D DCT (type-II) with orthonormal scaling."""
        # x: [H, W, Cin, Cout]
        # IDCT along W (dim=1)
        x_w = x.permute(0, 2, 3, 1).numpy()
        y_w = sfft.idct(x_w, axis=-1, type=2, norm='ortho')
        y = torch.from_numpy(y_w).permute(0, 3, 1, 2)
        # IDCT along H (dim=0)
        x_h = y.permute(1, 2, 3, 0).numpy()
        y_h = sfft.idct(x_h, axis=-1, type=2, norm='ortho')
        return torch.from_numpy(y_h).permute(3, 0, 1, 2)
    def create_mask(self, h: int, w: int, lo: float, hi: float) -> np.ndarray:
        u = (np.arange(h) / (h - 1)) if h > 1 else np.zeros(h)
        v = (np.arange(w) / (w - 1)) if w > 1 else np.zeros(w)
        U, V = np.meshgrid(u, v, indexing='ij')
        r = np.sqrt(U**2 + V**2) / np.sqrt(2.0)
        return ((r >= lo) & (r < hi)).astype(np.float32)
    band_defs = {"low": (0.0, 0.25), "mid": (0.25, 0.5), "high": (0.5, 1.0)}
    def band_energies_from_kernel(self, kernel_np: np.ndarray):
        """Return band energies, ratios, and totals for a Conv2d kernel."""
        h, w, _, cout = kernel_np.shape
        X = self.dct2_ortho(torch.from_numpy(kernel_np)).numpy()  # [H,W,Cin,Cout]
        totals = np.zeros((cout,), np.float64)
        band_E: Dict[str, np.ndarray] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(h, w, lo, hi)
            mask = mask[..., np.newaxis, np.newaxis]
            coeffs = X * mask
            energy = np.sqrt(np.sum(np.square(coeffs), axis=(0, 1, 2)))
            band_E[band] = energy
            totals += energy
        eps = 1e-12
        ratios = {band: band_E[band] / (totals + eps) for band in band_E}
        return band_E, ratios, totals
    def band_grad_energies_from_batch(
        self,
        model: nn.Module,
        layer: nn.Conv2d,
        xb: torch.Tensor,
        yb: torch.Tensor,
        from_logits: bool,
    ) -> Dict[str, np.ndarray]:
        """Gradient-based Taylor energies split per frequency band for one batch."""
        device = xb.device
        sparse = (yb.dim() == 1) or (yb.dim() == 2 and yb.size(-1) == 1)
        loss_fn = nn.CrossEntropyLoss()
        preds = model(xb)
        target = torch.argmax(yb, dim=1) if not sparse else yb
        loss = loss_fn(preds, target)
        grads = torch.autograd.grad(loss, layer.weight, create_graph=False)[0]
        if grads is None:
            return {band: np.zeros((layer.out_channels,), np.float64) for band in self.band_defs}
        G = self.dct2_ortho(grads).numpy()
        band_G: Dict[str, np.ndarray] = {}
        for band, (lo, hi) in self.band_defs.items():
            mask = self.create_mask(G.shape[0], G.shape[1], lo, hi)
            mask = mask[..., np.newaxis, np.newaxis]
            coeffs = G * mask
            energy = np.sqrt(np.sum(np.square(coeffs), axis=(0, 1, 2)))
            band_G[band] = energy
        return band_G
    # ---------------------------- dataset builders ---------------------------- #
    def build_frn_training_set(
        self,
        model: nn.Module,
        dataset: DataLoader,
        max_batches: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy smaller builder using only frequency ratios."""
        from_logits = True  # assume logits
        binary_targets = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if binary_targets else 3
        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)
        device = next(model.parameters()).device
        grad_acc: Dict[str, Dict[str, np.ndarray]] = {str(id(l)): None for l in conv_layers}
        model.train()
        for batch_idx, (xb, yb) in enumerate(dataset):
            if batch_idx >= max_batches:
                break
            xb = xb.to(device)
            for l_idx, layer in enumerate(conv_layers):
                layer_id = str(id(layer))
                band_grads = self.band_grad_energies_from_batch(model, layer, xb, yb, from_logits)
                if grad_acc[layer_id] is None:
                    grad_acc[layer_id] = {band: val.copy() for band, val in band_grads.items()}
                else:
                    for band in self.band_defs:
                        grad_acc[layer_id][band] += band_grads[band]
        model.eval()
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        for layer in conv_layers:
            kernel = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)
            band_E, ratios, _ = self.band_energies_from_kernel(kernel)
            cout = kernel.shape[-1]
            grad_band = grad_acc[str(id(layer))] or {
                band: np.zeros((cout,), np.float64) for band in self.band_defs
            }
            grad_tot = np.zeros((cout,), np.float64)
            for band in self.band_defs:
                grad_tot += grad_band[band]
            eps = 1e-12
            Y = np.stack(
                [
                    grad_band["low"] / (grad_tot + eps),
                    grad_band["mid"] / (grad_tot + eps),
                    grad_band["high"] / (grad_tot + eps),
                ],
                axis=1,
            )
            if binary_targets:
                low_col = Y[:, 0:1]
                other_col = np.sum(Y[:, 1:], axis=1, keepdims=True)
                Y = np.concatenate([low_col, other_col], axis=1)
                Y = Y / (np.sum(Y, axis=1, keepdims=True) + eps)
            X = np.stack(
                [ratios["low"], ratios["mid"], ratios["high"]],
                axis=1,
            )
            mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
            if not np.any(mask):
                continue
            X_list.append(X[mask])
            Y_list.append(Y[mask])
        if not X_list:
            return np.zeros((0, num_classes), np.float32), np.zeros((0, num_classes), np.float32)
        X_all = np.concatenate(X_list, axis=0).astype(np.float32)
        Y_all = np.concatenate(Y_list, axis=0).astype(np.float32)
        return X_all, Y_all
    def build_frn_training_set_stronger(
        self,
        model: nn.Module,
        dataset: DataLoader,
        band_defs: Dict[str, Tuple[float, float]],
        max_batches: int = 200,
        ema_beta: float = 0.8,
        sharpen_gamma: float = 2.0,
        eps: float = 1e-12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce FRN training data enriched with activation statistics.
        Returns:
            X_all: [N, F] features (band ratios + activation stats).
            Y_all: [N, C] normalized Taylor-based targets.
            W_all: [N] sample weights for band balancing.
        """
        binary_targets = bool(getattr(self.config, "frn_low_vs_rest", False))
        num_classes = 2 if binary_targets else 3
        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
            )
        device = next(model.parameters()).device
        layer_masks: Dict[str, Dict[str, np.ndarray]] = {}
        taylor_acc: Dict[str, Dict[str, np.ndarray]] = {}
        energy_acc: Dict[str, Dict[str, np.ndarray]] = {}
        totals_energy: Dict[str, np.ndarray] = {}
        act_abs_acc: Dict[str, np.ndarray] = {}
        act_sq_acc: Dict[str, np.ndarray] = {}
        act_count: Dict[str, float] = {}
        for layer in conv_layers:
            layer_id = str(id(layer))
            kernel = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # (H, W, Cin, Cout)
            h, w, _, cout = kernel.shape
            mask_dict: Dict[str, np.ndarray] = {}
            for band, (lo, hi) in band_defs.items():
                mask = self.create_mask(h, w, lo, hi)
                mask_dict[band] = mask[..., np.newaxis, np.newaxis]
            layer_masks[layer_id] = mask_dict
            kernel_np = kernel
            kernel_dct = self.dct2_ortho(torch.from_numpy(kernel_np)).numpy()
            taylor_acc[layer_id] = {
                band: np.zeros((cout,), np.float64) for band in band_defs
            }
            energy_acc[layer_id] = {}
            totals_energy[layer_id] = np.zeros((cout,), np.float64)
            for band, mask in mask_dict.items():
                coeffs = kernel_dct * mask
                energy = np.sqrt(np.sum(np.square(coeffs), axis=(0, 1, 2)))
                energy_acc[layer_id][band] = energy
                totals_energy[layer_id] += energy
            act_abs_acc[layer_id] = np.zeros((cout,), np.float64)
            act_sq_acc[layer_id] = np.zeros((cout,), np.float64)
            act_count[layer_id] = 0.0
        loss_fn = nn.CrossEntropyLoss()
        batches = 0
        model.train()
        for batch_idx, (xb, yb) in enumerate(dataset):
            if batches >= max_batches:
                break
            xb = xb.to(device)
            sparse = (yb.dim() == 1) or (yb.dim() == 2 and yb.size(-1) == 1)
            activations = []
            hooks = []
            def hook_fn(m, i, o):
                activations.append(o.detach())
            for l in conv_layers:
                hooks.append(l.register_forward_hook(hook_fn))
            preds = model(xb)
            for h in hooks:
                h.remove()
            if sparse:
                y_true = yb.to(device)
            else:
                y_true = torch.argmax(yb, dim=1).to(device)
            loss = loss_fn(preds, y_true)
            grads = torch.autograd.grad(loss, [l.weight for l in conv_layers], create_graph=False, allow_unused=True)
            for l_idx, layer in enumerate(conv_layers):
                layer_id = str(id(layer))
                act = activations[l_idx].cpu().numpy()
                abs_sum = np.sum(np.abs(act), axis=(0, 2, 3))
                sq_sum = np.sum(np.square(act), axis=(0, 2, 3))
                count = float(act.shape[0] * act.shape[2] * act.shape[3])
                act_abs_acc[layer_id] += abs_sum.astype(np.float64)
                act_sq_acc[layer_id] += sq_sum.astype(np.float64)
                act_count[layer_id] += count
                grad = grads[l_idx]
                if grad is None:
                    continue
                grad = grad.cpu()  # Move grad to CPU to match kernel's device
                kernel = layer.weight.detach().cpu()
                taylor = torch.abs(kernel * grad).permute(2, 3, 1, 0).numpy()
                taylor_dct = self.dct2_ortho(torch.from_numpy(taylor)).numpy()
                for band, mask in layer_masks[layer_id].items():
                    coeffs = taylor_dct * mask
                    energy = np.sqrt(np.sum(np.square(coeffs), axis=(0, 1, 2)))
                    if ema_beta > 0:
                        taylor_acc[layer_id][band] = (
                            ema_beta * taylor_acc[layer_id][band]
                            + (1.0 - ema_beta) * energy
                        )
                    else:
                        taylor_acc[layer_id][band] += energy
            batches += 1
        model.eval()
        if batches == 0:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
            )
        def _normalize(arr: np.ndarray) -> np.ndarray:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std < 1e-12:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - mean) / (std + 1e-12)
        X_list: List[np.ndarray] = []
        Y_list: List[np.ndarray] = []
        argmax_list: List[np.ndarray] = []
        for layer in conv_layers:
            layer_id = str(id(layer))
            cout = totals_energy[layer_id].shape[0]
            total_energy = totals_energy[layer_id] + eps
            ratios = {
                band: energy_acc[layer_id][band] / total_energy
                for band in band_defs
            }
            taylor_tot = np.zeros((cout,), np.float64)
            for band in band_defs:
                taylor_tot += taylor_acc[layer_id][band]
            targets = np.stack(
                [
                    np.power(taylor_acc[layer_id]["low"] + eps, 1.0 / sharpen_gamma),
                    np.power(taylor_acc[layer_id]["mid"] + eps, 1.0 / sharpen_gamma),
                    np.power(taylor_acc[layer_id]["high"] + eps, 1.0 / sharpen_gamma),
                ],
                axis=1,
            )
            targets = targets / (np.sum(targets, axis=1, keepdims=True) + eps)
            if binary_targets:
                low_col = targets[:, 0:1]
                other_col = np.sum(targets[:, 1:], axis=1, keepdims=True)
                combined = np.concatenate([low_col, other_col], axis=1)
                targets = combined / (np.sum(combined, axis=1, keepdims=True) + eps)
            count = max(act_count[layer_id], eps)
            mean_abs = act_abs_acc[layer_id] / count
            mean_sq = act_sq_acc[layer_id] / count
            variance = np.maximum(mean_sq - np.square(mean_abs), 0.0)
            std = np.sqrt(variance)
            mean_abs_norm = _normalize(mean_abs)
            std_norm = _normalize(std)
            features = np.stack(
                [ratios["low"], ratios["mid"], ratios["high"]],
                axis=1,
            )
            if bool(getattr(self.config, "frn_use_activation_features", True)):
                extra = np.stack([mean_abs_norm, std_norm], axis=1)
                features = np.concatenate([features, extra], axis=1)
            mask = np.isfinite(features).all(axis=1) & np.isfinite(targets).all(axis=1)
            if not np.any(mask):
                continue
            X_list.append(features[mask].astype(np.float32))
            Y_list.append(targets[mask].astype(np.float32))
            argmax_list.append(np.argmax(targets[mask], axis=1))
        if not X_list:
            return (
                np.zeros((0, 3), np.float32),
                np.zeros((0, 3), np.float32),
                np.zeros((0,), np.float32),
            )
        X_all = np.concatenate(X_list, axis=0)
        Y_all = np.concatenate(Y_list, axis=0)
        argmax_all = np.concatenate(argmax_list, axis=0)
        counts = np.bincount(argmax_all, minlength=num_classes).astype(np.float32)
        inv_freq = 1.0 / (counts + 1e-6)
        inv_freq = inv_freq / np.sum(inv_freq) * 3.0
        w_all = inv_freq[argmax_all]
        max_weight = float(getattr(self.config, "frn_weight_clip", 3.0))
        if max_weight > 0:
            w_all = np.minimum(w_all, max_weight)
        return X_all.astype(np.float32), Y_all.astype(np.float32), w_all.astype(np.float32)