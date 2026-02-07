import os
import torch.nn as nn
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import random
import numpy as np
import torch
from pruning.frequency_rel_analyzer import FrequencyRelevanceAnalyzer
from pruning.pruning_strategy import PruningStrategy, LayerPruningConfig
from models.model_utils import ModelUtils
from utils.logger import Logger
from training.frequency_regularizer import compute_spectral_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class HybridFrequencyBaseline:
    """
    Hybrid frequency-saliency baseline that combines DCT-based energy analysis
    with gradient saliency for iterative channel pruning.
    """
    def __init__(self, config, trainer, evaluator):
        self.config = config
        self.trainer = trainer
        self.evaluator = evaluator
        self.logger = Logger(config)
        self.pruning_util = PruningStrategy(config)
        self.frn_analyzer = FrequencyRelevanceAnalyzer(config)
        self.frequency_layer_names: List[str] = []
        self.entropy_targets: Dict[str, float] = {}
        mode = str(getattr(self.config, "hybrid_mode", "frequency")).lower()
        self.frequency_enabled = mode == "frequency"
    # --------------------------- public API --------------------------- #
    def run_pipeline(
        self,
        model: torch.nn.Module,
        train_dl: torch.utils.data.DataLoader,
        val_dl: torch.utils.data.DataLoader,
        train_eval_dl: Optional[torch.utils.data.DataLoader] = None,
        activation_dl: Optional[torch.utils.data.DataLoader] = None,
    ) -> Tuple[torch.nn.Module, List[Dict]]:
        """
        Execute the iterative pruning baseline.
        Returns:
            pruned_model: Model after pruning iterations.
            iteration_history: metrics per iteration.
        """
        activation_dl = activation_dl or train_eval_dl or train_dl
        frn_model = self._train_frn(model, activation_dl)
        mode_label = "frequency-regularized" if self.frequency_enabled else "original"
        self.logger.info(f"Hybrid baseline mode: {mode_label}")
        self.frequency_layer_names = []
        self.entropy_targets = {}
        if self.frequency_enabled:
            self.frequency_layer_names = self._select_frequency_layers(model)
            if activation_dl is not None and self.frequency_layer_names:
                self.entropy_targets = self._estimate_entropy_targets(
                    model,
                    activation_dl,
                    self.frequency_layer_names,
                    batches=getattr(self.config, "frequency_entropy_target_batches", 8),
                )
                self.logger.info(
                    f"Captured spectral entropy targets for {len(self.entropy_targets)} layer(s)."
                )
        baseline_metrics = self.evaluator.evaluate_model(model, val_dl, "Hybrid Baseline (validation)")
        baseline_loss = float(baseline_metrics.get("loss", 0.0))
        baseline_acc = baseline_metrics.get("accuracy")
        baseline_acc_str = f"{baseline_acc:.4f}" if baseline_acc is not None else "nan"
        self.logger.info(
            f"Hybrid baseline: initial val_loss={baseline_loss:.4f}, "
            f"val_accuracy={baseline_acc_str}"
        )
        iteration = 0
        iteration_history: List[Dict] = []
        kappa_ratio = float(self.config.hybrid_initial_kappa_ratio)
        current_model = model
        target_refresh = int(getattr(self.config, "frequency_entropy_refresh_interval", 0))
        while iteration < self.config.hybrid_iterations:
            iteration += 1
            self.logger.info(f"Hybrid baseline iteration {iteration}/{self.config.hybrid_iterations}...")
            if self.frequency_enabled and target_refresh > 0 and (iteration - 1) % target_refresh == 0 and iteration > 1:
                self._refresh_entropy_targets(current_model, activation_dl)
            activation_stats = None
            use_activation = bool(getattr(self.config, "frn_use_activation_features", True))
            if frn_model is not None and activation_dl is not None and use_activation:
                activation_stats = self._compute_activation_statistics(
                    current_model,
                    activation_dl,
                    max_batches=int(getattr(self.config, "frn_activation_batches", 8)),
                )
            freq_scores = self._compute_frequency_scores(
                current_model,
                kappa_ratio,
                frn_model,
                activation_stats=activation_stats,
            )
            grad_scores = self._compute_gradient_saliency(current_model, train_eval_dl or train_dl)
            hybrid_scores = self._combine_scores(freq_scores, grad_scores)
            masks = self._select_pruning_masks(hybrid_scores, iteration)
            # if "conv1" in masks:
            # masks["conv1"] = np.ones_like(masks["conv1"], dtype=bool)
            current_model = self._apply_pruning(current_model, masks, hybrid_scores)
            # self._validate_and_recalibrate(current_model, train_eval_dl or train_dl)
            # if self.frequency_enabled:
            # self._frequency_balanced_regrow_init(current_model, iteration)
            warmup_epochs = max(0, int(getattr(self.config, "hybrid_warmup_epochs", 0)))
            if warmup_epochs > 0:
                self.logger.info(f"Warm-up training for {warmup_epochs} epoch(s) before fine-tuning...")
                self.trainer.compile_model(
                    current_model,
                    learning_rate=self.config.hybrid_warmup_lr,
                )
                self.trainer.train_cifar(
                    current_model,
                    train_dataset=train_dl,
                    val_dataset=val_dl,
                    epochs=warmup_epochs,
                    train_eval_dl=train_eval_dl,
                )
            freq_reg_config = None
            teacher = None
            if self.frequency_enabled and iteration > 1:
                freq_reg_config = self._build_frequency_regularizer_config()
                teacher = model
            current_model, _ = self.trainer.fine_tune_cifar(
                current_model,
                train_dl,
                val_dl,
                epochs=self.config.hybrid_finetune_epochs,
                learning_rate=self.config.pruned_growth_lr,
                log_dir_suffix=f"hybrid_iter_{iteration}",
                train_eval_loader=train_eval_dl,
                # frequency_regularizer_config=freq_reg_config,
                # teacher_model=teacher,
            )
            metrics = self.evaluator.evaluate_model(current_model, val_dl, "Hybrid Baseline (validation)")
            val_acc = metrics.get("accuracy")
            val_loss = float(metrics.get("loss", 0.0))
            param_stats = ModelUtils.count_parameters(current_model)
            flop_count = ModelUtils.compute_flops(current_model, self.config.input_shape_cifar10)
            iteration_history.append(
                {
                    "iteration": iteration,
                    "kappa_ratio": kappa_ratio,
                    "metrics": metrics,
                    "param_count": param_stats,
                    "flops": flop_count,
                }
            )
            val_acc_str = f"{val_acc:.4f}" if val_acc is not None else "nan"
            self.logger.info(
                f"Iteration {iteration} summary -> val_acc={val_acc_str}, "
                f"params={param_stats['total']:,}, FLOPs={flop_count/1e6:.2f} MFLOPs"
            )
            delta_loss = max(0.0, val_loss - baseline_loss)
            kappa_ratio = max(
                0.05,
                kappa_ratio * (1.0 - self.config.hybrid_kappa_beta * delta_loss),
            )
            baseline_loss = val_loss
        return current_model, iteration_history
    def _select_frequency_layers(self, model: torch.nn.Module) -> List[str]:
        conv_layers = ModelUtils.get_conv_layers(model)
        max_layers = max(0, int(getattr(self.config, "frequency_regularization_layers", 0)))
        if not conv_layers or max_layers <= 0:
            return []
        return [name for name, _ in list(model.named_modules())[-max_layers:] if isinstance(_, nn.Conv2d)]
    def _estimate_entropy_targets(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        layer_names: List[str],
        batches: int = 8,
    ) -> Dict[str, float]:
        if dataloader is None or not layer_names:
            return {}
        device = next(model.parameters()).device
        hooks = []
        activations = {}
        def hook_fn(name):
            def hook(m, i, o):
                activations[name] = o.detach()
            return hook
        for name in layer_names:
            module = dict(model.named_modules())[name]
            hooks.append(module.register_forward_hook(hook_fn(name)))
        stats: Dict[str, List[float]] = {name: [] for name in layer_names}
        model.eval()
        with torch.no_grad():
            for batch_idx, (xb, _) in enumerate(dataloader):
                if batch_idx >= batches:
                    break
                xb = xb.to(device)
                _ = model(xb)
                for name in layer_names:
                    feat = activations.get(name)
                    if feat is None:
                        continue
                    entropy = compute_spectral_entropy(feat)
                    stats[name].append(float(entropy.item()))
        for h in hooks:
            h.remove()
        return {
            name: float(np.mean(values)) for name, values in stats.items() if values
        }
    def _refresh_entropy_targets(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
        batches: Optional[int] = None,
    ) -> None:
        if dataloader is None or not self.frequency_layer_names:
            return
        batch_count = batches or getattr(self.config, "frequency_entropy_target_batches", 8)
        self.entropy_targets = self._estimate_entropy_targets(
            model,
            dataloader,
            self.frequency_layer_names,
            batches=batch_count,
        )
        self.logger.info(
            f"Refreshed spectral entropy targets for {len(self.entropy_targets)} layer(s)."
        )
    def _build_frequency_regularizer_config(self) -> Optional[Dict[str, object]]:
        if not self.frequency_layer_names or not self.entropy_targets:
            return None
        ordered_targets = [
            (name, self.entropy_targets.get(name)) for name in self.frequency_layer_names
        ]
        ordered_targets = [(n, t) for n, t in ordered_targets if t is not None]
        if not ordered_targets:
            return None
        beta = float(getattr(self.config, "frequency_entropy_beta", 0.05))
        layer_weights = getattr(self.config, "frequency_entropy_layer_weights", {}) or {}
        if not layer_weights:
            num_layers = len(ordered_targets)
            if num_layers:
                scales = np.linspace(1.0, 0.4, num_layers)
                layer_weights = {
                    name: float(scale) for (name, _), scale in zip(ordered_targets, scales)
                }
        targets = {name: target for name, target in ordered_targets}
        return {
            "layer_names": list(targets.keys()),
            "targets": targets,
            "beta": beta,
            "layer_weights": layer_weights,
        }
    def _frequency_balanced_regrow_init(self, model: torch.nn.Module, iteration: int) -> None:
        regrow_fraction = float(getattr(self.config, "hybrid_regrow_fraction", 0.0))
        if regrow_fraction <= 0.0:
            return
        if iteration <= 1:
            self.logger.debug("Regrow skipped on first pruning iteration to preserve baseline weights.")
            return
        conv_layers = ModelUtils.get_conv_layers(model)
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            kernel = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # (H, W, Cin, Cout)
            h, w, cin, cout = kernel.shape
            if cout == 0:
                continue
            regrow_count = max(1, int(round(cout * regrow_fraction)))
            regrow_count = min(regrow_count, cout)
            dct_kernel = self._dct2(kernel)
            band_ratios = self._band_ratios(dct_kernel, self.config.frequency_bands)
            if band_ratios.size == 0:
                continue
            band_names = list(self.config.frequency_bands.keys())
            band_energy = band_ratios.mean(axis=0)
            target_idx = int(np.argmin(band_energy))
            band_name = band_names[target_idx]
            lo, hi = self.config.frequency_bands[band_name]
            norms = np.linalg.norm(kernel.reshape(-1, cout), axis=0)
            regrow_indices = np.argsort(norms)[:regrow_count]
            mask = self.frn_analyzer.create_mask(h, w, lo, hi)
            freq_noise = np.random.normal(size=(h, w, cin, regrow_count), scale=0.05)
            freq_noise *= mask[..., np.newaxis, np.newaxis]
            spatial_kernels = self._idct2(freq_noise)
            new_kernel = kernel.copy()
            for slot, filt_idx in enumerate(regrow_indices):
                new_kernel[:, :, :, filt_idx] = spatial_kernels[:, :, :, slot]
                if layer.bias is not None:
                    layer.bias.data[filt_idx] = 0.0
            layer.weight.data = torch.from_numpy(new_kernel.transpose(3, 2, 0, 1)).to(layer.weight.device)
            self.logger.debug(
                f"Regrew {len(regrow_indices)} filter(s) in {name} targeting {band_name} band."
            )
    def _validate_and_recalibrate(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
    ) -> None:
        try:
            input_shape = self.config.input_shape_cifar10
            dummy = torch.zeros((1, ) + input_shape, dtype=torch.float32, device=next(model.parameters()).device)
            model(dummy)
        except Exception as exc:
            self.logger.warning("Failed dummy forward after pruning: %s", exc)
        if dataloader is None:
            return
        steps = int(getattr(self.config, "bn_recalibrate_steps", 200))
        steps = max(1, steps)
        model.train()
        for step, (xb, _) in enumerate(dataloader):
            if step >= steps:
                break
            xb = xb.to(next(model.parameters()).device)
            _ = model(xb)
        model.eval()
    # --------------------------- FRN helpers --------------------------- #
    def _train_frn(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
    ) -> Optional[torch.nn.Module]:
        if dataloader is None:
            self.logger.info("FRN training skipped: no dataloader provided.")
            return None
        seed_value = getattr(self.config, "seed", None)
        if seed_value is None:
            seed_value = 12345
        seed_int = int(seed_value)
        np.random.seed(seed_int % (2**32))
        random.seed(seed_int % (2**32))
        torch.manual_seed(seed_int)
        self.logger.info("Building FRN training set for hybrid baseline...")
        X_all, Y_all, W_all = self.frn_analyzer.build_frn_training_set_stronger(
            model=model,
            dataset=dataloader,
            band_defs=self.config.frequency_bands,
            max_batches=256,
            ema_beta=getattr(self.config, "frn_ema_beta", 0.8),
            sharpen_gamma=getattr(self.config, "frn_sharpen_gamma", 2.0),
        )
        if X_all.size == 0 or Y_all.size == 0:
            self.logger.info("FRN dataset empty; skipping FRN training.")
            return None
        if W_all.size:
            W_all = W_all / (np.mean(W_all) + 1e-12)
        available_features = int(X_all.shape[1])
        desired_features = getattr(self.config, "frn_feature_count", None)
        if desired_features is None or desired_features <= 0:
            input_dim = available_features
        else:
            input_dim = max(1, min(int(desired_features), available_features))
        features = X_all[:, :input_dim]
        val_fraction = float(getattr(self.config, "frn_validation_split", 0.2))
        val_fraction = min(max(val_fraction, 0.0), 0.5)
        min_val_samples = int(getattr(self.config, "frn_min_validation_samples", 64))
        min_val_samples = max(0, min_val_samples)
        n_samples = features.shape[0]
        val_target = 0
        if n_samples >= 4:
            val_target = int(round(n_samples * val_fraction))
            if n_samples > min_val_samples:
                val_target = max(val_target, min_val_samples)
            val_target = min(val_target, max(0, n_samples - 1))
        rng = np.random.default_rng(seed=seed_int)
        labels = np.argmax(Y_all, axis=1)
        unique_classes = np.unique(labels)
        class_indices: Dict[int, np.ndarray] = {}
        for cls in unique_classes:
            idx = np.where(labels == cls)[0]
            if idx.size == 0:
                continue
            class_indices[int(cls)] = rng.permutation(idx)
        val_splits: Dict[int, np.ndarray] = {}
        train_splits: Dict[int, np.ndarray] = {}
        remaining_target = val_target
        for cls, idx in class_indices.items():
            count = idx.size
            if count == 0:
                val_splits[cls] = np.empty((0,), dtype=int)
                train_splits[cls] = np.empty((0,), dtype=int)
                continue
            if val_target <= 0:
                take = 0
            else:
                base = int(round(count * val_fraction))
                base = max(1, base) if count > 1 else min(base, count)
                take = min(base, count - 1) if count > 1 else min(base, count)
            take = max(0, min(take, count))
            val_splits[cls] = idx[:take]
            train_splits[cls] = idx[take:]
            remaining_target -= take
        if remaining_target > 0:
            for cls, idx in class_indices.items():
                if remaining_target <= 0:
                    break
                train_part = train_splits.get(cls, np.empty((0,), dtype=int))
                if train_part.size == 0:
                    continue
                take = min(remaining_target, train_part.size)
                extra = train_part[:take]
                val_splits[cls] = np.concatenate([val_splits.get(cls, np.empty((0,), dtype=int)), extra])
                train_splits[cls] = train_part[take:]
                remaining_target -= take
        total_val = sum(split.size for split in val_splits.values())
        if total_val > val_target > 0:
            surplus = total_val - val_target
            classes_sorted = sorted(val_splits.items(), key=lambda item: item[1].size, reverse=True)
            for cls, val_idx in classes_sorted:
                if surplus <= 0:
                    break
                if val_idx.size <= 1:
                    continue
                give_back = min(surplus, val_idx.size - 1)
                if give_back <= 0:
                    continue
                keep = val_idx[:-give_back]
                move = val_idx[-give_back:]
                val_splits[cls] = keep
                train_splits[cls] = np.concatenate([train_splits.get(cls, np.empty((0,), dtype=int)), move])
                surplus -= give_back
        val_indices = np.concatenate([arr for arr in val_splits.values() if arr.size > 0]) if val_splits else np.empty((0,), dtype=int)
        train_indices = np.concatenate([arr for arr in train_splits.values() if arr.size > 0]) if train_splits else np.arange(n_samples)
        if val_target <= 0 or val_indices.size == 0:
            val_indices = np.empty((0,), dtype=int)
            train_indices = np.arange(n_samples)
        X_train = features[train_indices]
        Y_train = Y_all[train_indices]
        W_train = W_all[train_indices]
        if val_indices.size > 0:
            X_val = features[val_indices]
            Y_val = Y_all[val_indices]
            W_val = W_all[val_indices]
        else:
            X_val = Y_val = W_val = None
        if W_train.size:
            W_train = W_train / (np.mean(W_train) + 1e-12)
        if W_val is not None and W_val.size:
            W_val = W_val / (np.mean(W_val) + 1e-12)
        hidden_units_cfg = getattr(self.config, "frn_hidden_units", (64, 32))
        if isinstance(hidden_units_cfg, (int, float)):
            hidden_units = (int(hidden_units_cfg),)
        elif isinstance(hidden_units_cfg, (list, tuple)):
            hidden_units = tuple(int(u) for u in hidden_units_cfg if int(u) > 0)
        else:
            hidden_units = (64, 32)
        if not hidden_units:
            hidden_units = (64, 32)
        num_classes = 2 if bool(getattr(self.config, "frn_low_vs_rest", False)) else 3
        frn_model = self.frn_analyzer.build_frequency_relevance_net(
            hidden_units=hidden_units,
            input_dim=input_dim,
            architecture=getattr(self.config, "frn_architecture", "residual"),
            num_classes=num_classes,
        ).to(device)
        frn_epochs = int(max(1, getattr(self.config, "frn_epochs", 20)))
        frn_batch_size = int(max(1, getattr(self.config, "frn_batch_size", 256)))
        frn_initial_lr = float(getattr(self.config, "frn_initial_lr", 3e-4))
        frn_min_lr = float(getattr(self.config, "frn_min_lr", 1e-5))
        cosine_floor = float(getattr(self.config, "frn_cosine_min_factor", 0.1))
        steps_per_epoch = max(1, int(np.ceil(max(1, X_train.shape[0]) / frn_batch_size)))
        decay_steps = frn_epochs * steps_per_epoch
        if frn_initial_lr <= 0.0:
            frn_initial_lr = 3e-4
        if frn_min_lr <= 0.0:
            frn_min_lr = 1e-6
        alpha = max(0.0, min(1.0, max(cosine_floor, frn_min_lr / frn_initial_lr)))
        # PyTorch cosine decay
        optimizer = torch.optim.AdamW(frn_model.parameters(), lr=frn_initial_lr, weight_decay=2e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=frn_epochs, eta_min=frn_initial_lr * alpha)
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.logger.info(
            f"Training FRN ({n_samples} samples, input_dim={input_dim}, epochs={frn_epochs}, "
            f"val_split={val_fraction:.2f}, batch_size={frn_batch_size})"
        )
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train), torch.from_numpy(W_train))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=frn_batch_size, shuffle=True)
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val), torch.from_numpy(W_val))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=frn_batch_size, shuffle=False)
        else:
            val_loader = None
        frn_model.train()
        for epoch in range(frn_epochs):
            epoch_loss = 0.0
            for batch in train_loader:
                x_b, y_b, w_b = [t.to(device) for t in batch]
                optimizer.zero_grad()
                preds = frn_model(x_b)
                loss = loss_fn(preds, y_b.argmax(dim=1))
                weighted_loss = (loss * w_b).mean()
                weighted_loss.backward()
                optimizer.step()
                epoch_loss += weighted_loss.item()
            scheduler.step()
        frn_model.eval()
        # Evaluation
        train_loss = 0.0
        with torch.no_grad():
            for batch in train_loader:
                x_b, y_b, w_b = [t.to(device) for t in batch]
                preds = frn_model(x_b)
                loss = loss_fn(preds, y_b.argmax(dim=1))
                weighted_loss = (loss * w_b).mean()
                train_loss += weighted_loss.item()
        train_loss /= len(train_loader)
        if val_loader:
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_b, y_b, w_b = [t.to(device) for t in batch]
                    preds = frn_model(x_b)
                    loss = loss_fn(preds, y_b.argmax(dim=1))
                    weighted_loss = (loss * w_b).mean()
                    val_loss += weighted_loss.item()
            val_loss /= len(val_loader)
            msg = f"FRN training complete | loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        else:
            msg = f"FRN training complete | loss={train_loss:.4f}"
        self.logger.info(msg)
        if getattr(self.config, "frn_plot_training", False):
            # Plotting code would need adaptation, skipping for brevity
            pass
        return frn_model
    # --------------------------- scoring --------------------------- #
    def _compute_frequency_scores(
        self,
        model: torch.nn.Module,
        kappa_ratio: float,
        frn_model: Optional[torch.nn.Module],
        activation_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> Dict[str, np.ndarray]:
        scores: Dict[str, np.ndarray] = {}
        frequency_bands = self.config.frequency_bands
        activation_stats = activation_stats or {}
        device = next(frn_model.parameters()).device if frn_model else torch.device('cpu')
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            kernel = layer.weight.detach().cpu().numpy().transpose(2, 3, 1, 0)  # (H, W, Cin, Cout)
            dct_kernel = self._dct2(kernel)
            band_ratios = self._band_ratios(dct_kernel, frequency_bands)
            low_ratio = self._low_frequency_ratio(dct_kernel, kappa_ratio)
            freq_score = low_ratio
            if frn_model is not None and band_ratios.size:
                input_dim = frn_model.fc1.in_features if hasattr(frn_model, 'fc1') else 3
                feature_parts: List[np.ndarray] = [band_ratios.astype(np.float32)]
                extra = activation_stats.get(name)
                if extra is not None:
                    mean_abs_norm, std_norm = extra
                    feature_parts.append(mean_abs_norm[:, None])
                    feature_parts.append(std_norm[:, None])
                features = np.concatenate(feature_parts, axis=1) if len(feature_parts) > 1 else feature_parts[0]
                if input_dim and features.shape[1] != input_dim:
                    if features.shape[1] > input_dim:
                        features = features[:, :input_dim]
                    else:
                        pad_width = input_dim - features.shape[1]
                        pad = np.zeros((features.shape[0], pad_width), dtype=features.dtype)
                        features = np.concatenate([features, pad], axis=1)
                features_t = torch.from_numpy(features).to(device)
                with torch.no_grad():
                    preds = frn_model(features_t)
                low_weight = preds[:, 0].cpu().numpy()
                freq_score = low_ratio * (1.0 + low_weight)
            scores[name] = freq_score
        return scores
    def _compute_gradient_saliency(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int = 16,
    ) -> Dict[str, np.ndarray]:
        loss_fn = nn.CrossEntropyLoss()
        conv_layers = ModelUtils.get_conv_layers(model)
        kernels = [l.weight for l in conv_layers]
        saliency = {str(id(l)): np.zeros(l.out_channels, dtype=np.float32) for l in conv_layers}
        device = next(model.parameters()).device
        model.train()
        for batch_idx, (xb, yb) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            xb = xb.to(device)
            yb = torch.argmax(yb, dim=1).to(device) if yb.dim() == 2 else yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            grads = torch.autograd.grad(loss, kernels, allow_unused=True)
            for l_idx, grad in enumerate(grads):
                layer_id = str(id(conv_layers[l_idx]))
                if grad is None:
                    continue
                kernel = conv_layers[l_idx].weight.detach()
                taylor = torch.abs(grad * kernel).cpu().numpy()
                per_filter = np.sum(taylor, axis=(1, 2, 3))
                saliency[layer_id] += per_filter.astype(np.float32)
        model.eval()
        saliency_named = {name: saliency[str(id(layer))] for name, layer in model.named_modules() if isinstance(layer, nn.Conv2d)}
        return saliency_named
    def _combine_scores(
        self,
        frequency_scores: Dict[str, np.ndarray],
        gradient_scores: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        combined: Dict[str, np.ndarray] = {}
        alpha = float(self.config.hybrid_alpha)
        eps = 1e-8
        for layer_name, grad in gradient_scores.items():
            freq = frequency_scores.get(layer_name)
            if freq is None:
                combined[layer_name] = grad
                continue
            combined[layer_name] = grad * np.power(freq + eps, alpha)
        return combined
    def _select_pruning_masks_new(
        self,
        model: torch.nn.Module,
        scores: Dict[str, np.ndarray],
        iteration: int = 1,
    ) -> Dict[str, np.ndarray]:
        if not scores:
            return {}
        prune_ratio = float(self.config.hybrid_prune_fraction)
        taper_start = 6
        if iteration > taper_start:
            decay_steps = max(1, self.config.hybrid_iterations - taper_start)
            frac = (iteration - taper_start) / decay_steps
            target_floor = 0.02 if prune_ratio > 0.02 else prune_ratio
            prune_ratio = prune_ratio - (prune_ratio - target_floor) * min(1.0, frac)
        if prune_ratio <= 0.0:
            return {}
        normalized_scores: Dict[str, np.ndarray] = {}
        for layer_name, layer_scores in scores.items():
            if layer_scores.size == 0:
                continue
            mean = np.mean(layer_scores)
            std = np.std(layer_scores)
            if std < 1e-8:
                normalized_scores[layer_name] = layer_scores
            else:
                normalized_scores[layer_name] = (layer_scores - mean) / (std + 1e-8)
        if not normalized_scores:
            return {}
        structured_configs = self.pruning_util.select_filters_structured(
            normalized_scores,
            prune_ratio,
            model,
        )
        masks: Dict[str, np.ndarray] = {}
        for layer_name, config in structured_configs.items():
            masks[layer_name] = config.mask
        return masks
    def _select_pruning_masks(self, scores: Dict[str, np.ndarray], iteration: int = 1) -> Dict[str, np.ndarray]:
        prune_ratio = float(self.config.hybrid_prune_fraction)
        taper_start = 8
        if iteration > taper_start:
            decay_steps = max(1, self.config.hybrid_iterations - taper_start)
            frac = (iteration - taper_start) / decay_steps
            prune_ratio = prune_ratio - (prune_ratio - 0.05) * min(1.0, frac)
        pools: List[Tuple[str, int, float]] = []
        for layer_name, layer_scores in scores.items():
            for idx, value in enumerate(layer_scores):
                pools.append((layer_name, idx, float(value)))
        if not pools:
            return {}
        pools.sort(key=lambda item: item[2])
        total_filters = len(pools)
        target_prunes = int(total_filters * prune_ratio)
        if target_prunes <= 0:
            return {}
        prune_indices = {(layer, idx) for layer, idx, _ in pools[:target_prunes]}
        masks: Dict[str, np.ndarray] = {}
        for layer_name, layer_scores in scores.items():
            mask = np.ones_like(layer_scores, dtype=bool)
            for idx in range(len(layer_scores)):
                if (layer_name, idx) in prune_indices:
                    mask[idx] = False
            keep = int(mask.sum())
            if keep < self.config.hybrid_min_filters:
                # enforce minimum filter constraint by restoring highest scoring filters
                order = np.argsort(layer_scores)[::-1]
                mask[:] = False
                mask[order[:self.config.hybrid_min_filters]] = True
            masks[layer_name] = mask
        return masks
    # --------------------------- pruning --------------------------- #
    def _apply_pruning(
        self,
        model: torch.nn.Module,
        masks: Dict[str, np.ndarray],
        importance_scores: Optional[Dict[str, np.ndarray]] = None,
    ) -> torch.nn.Module:
        if not masks:
            return model
        importance_scores = importance_scores or {}
        configs: Dict[str, LayerPruningConfig] = {}
        conv_dict = {name: module for name, module in model.named_modules() if isinstance(module, nn.Conv2d)}
        for name, layer in conv_dict.items():
            mask = masks.get(name)
            if mask is None:
                continue
            layer_scores = importance_scores.get(name, np.ones_like(mask, dtype=np.float32))
            configs[name] = LayerPruningConfig(
                layer_name=name,
                original_filters=int(len(mask)),
                filters_to_keep=int(mask.sum()),
                pruning_ratio=float(1.0 - mask.mean()),
                importance_scores=layer_scores,
                mask=mask,
            )
        if not configs:
            return model
        pruned = self.pruning_util.apply_structured_pruning(model, configs)
        return pruned
    # --------------------------- utilities --------------------------- #
    def _dct2(self, kernel: np.ndarray) -> np.ndarray:
        return self.frn_analyzer.dct2_ortho(torch.from_numpy(kernel)).numpy()
    def _idct2(self, coeffs: np.ndarray) -> np.ndarray:
        return self.frn_analyzer.idct2_ortho(torch.from_numpy(coeffs)).numpy()
    def _band_ratios(
        self,
        dct_kernel: np.ndarray,
        bands: Dict[str, Tuple[float, float]],
    ) -> np.ndarray:
        h, w, _cin, cout = dct_kernel.shape
        ratios = []
        total_energy = np.sum(np.square(dct_kernel), axis=(0, 1, 2)) + 1e-8
        for name, (lo, hi) in bands.items():
            mask = self.frn_analyzer.create_mask(h, w, lo, hi)
            band_energy = np.sum(np.square(dct_kernel) * mask[:, :, None, None], axis=(0, 1, 2))
            ratios.append(band_energy / total_energy)
        if not ratios:
            return np.zeros((cout, 0), dtype=np.float32)
        stacked = np.stack(ratios, axis=1)
        return stacked.astype(np.float32)
    def _low_frequency_ratio(self, dct_kernel: np.ndarray, kappa_ratio: float) -> np.ndarray:
        h, w, _cin, _cout = dct_kernel.shape
        kappa = max(1, int(round(min(h, w) * kappa_ratio)))
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:kappa, :kappa] = 1.0
        total = np.sum(np.square(dct_kernel), axis=(0, 1, 2)) + 1e-8
        low = np.sum(np.square(dct_kernel) * mask[:, :, None, None], axis=(0, 1, 2))
        return (low / total).astype(np.float32)
    @staticmethod
    def _get_batch_inputs(batch):
        if isinstance(batch, (tuple, list)):
            return batch[0]
        return batch
    def _compute_activation_statistics(
        self,
        model: torch.nn.Module,
        dataloader: Optional[torch.utils.data.DataLoader],
        max_batches: int = 8,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        if dataloader is None or max_batches <= 0:
            return {}
        conv_layers = ModelUtils.get_conv_layers(model)
        if not conv_layers:
            return {}
        device = next(model.parameters()).device
        abs_acc: Dict[str, np.ndarray] = {}
        sq_acc: Dict[str, np.ndarray] = {}
        counts: Dict[str, float] = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            filters = layer.out_channels
            abs_acc[name] = np.zeros(filters, dtype=np.float64)
            sq_acc[name] = np.zeros(filters, dtype=np.float64)
            counts[name] = 0.0
        taken = 0
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                xb = self._get_batch_inputs(batch)
                xb = xb.to(device)
                activations = []
                hooks = []
                def hook_fn(m, i, o):
                    activations.append(o.detach().cpu().numpy())
                for l in conv_layers:
                    hooks.append(l.register_forward_hook(hook_fn))
                _ = model(xb)
                for h in hooks:
                    h.remove()
                for l_idx, act_np in enumerate(activations):
                    layer_name = [n for n, m in model.named_modules() if m is conv_layers[l_idx]][0]
                    if act_np.size == 0:
                        continue
                    abs_acc[layer_name] += np.sum(np.abs(act_np), axis=(0, 2, 3)).astype(np.float64)
                    sq_acc[layer_name] += np.sum(np.square(act_np), axis=(0, 2, 3)).astype(np.float64)
                    counts[layer_name] += float(act_np.shape[0] * act_np.shape[2] * act_np.shape[3])
                taken += 1
                if taken >= max_batches:
                    break
        if taken == 0:
            return {}
        def _normalize(arr: np.ndarray) -> np.ndarray:
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            if std < 1e-12:
                return np.zeros_like(arr, dtype=np.float64)
            return (arr - mean) / (std + 1e-12)
        stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name, layer in model.named_modules():
            if not isinstance(layer, nn.Conv2d):
                continue
            total = max(counts[name], 1e-12)
            mean_abs = abs_acc[name] / total
            mean_sq = sq_acc[name] / total
            variance = np.maximum(mean_sq - np.square(mean_abs), 0.0)
            std = np.sqrt(variance)
            stats[name] = (
                _normalize(mean_abs).astype(np.float32),
                _normalize(std).astype(np.float32),
            )
        return stats