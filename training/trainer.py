import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ExponentialLR, StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Callable, List, Tuple, Union
import os
from datetime import datetime
import json
import gc
import shutil
import numpy as np
from config import Config  # Assuming this is the same
from utils.logger import Logger  # Assuming this is the same
from utils.overfitting_monitor import OverfittingMonitor, AdaptiveRegularization  # Need to adapt if necessary
from training.distillation import Distiller  # Need to implement PyTorch version
from training.frequency_regularizer import (
    FrequencyRegularizedModel,
    SpectralEntropyRegularizer,
)  # Need to implement PyTorch version

class SplitTensorBoard:
    """Write train/val scalars into separate TensorBoard runs."""
    def __init__(self, train_dir: str, val_dir: str):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        self._train_writer = SummaryWriter(train_dir)
        self._val_writer = SummaryWriter(val_dir)

    def write_epoch(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if key.startswith('val_') or value is None:
                continue
            self._train_writer.add_scalar(key, float(value), epoch)
        for key, value in logs.items():
            if not key.startswith('val_') or value is None:
                continue
            tag = key[len('val_'):]
            self._val_writer.add_scalar(tag, float(value), epoch)
        self._train_writer.flush()
        self._val_writer.flush()

    def close(self):
        self._train_writer.close()
        self._val_writer.close()

class HSGSPTrainer:
    """Training manager for HSGSP pruning"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config)
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    @staticmethod
    def _model_outputs_logits(model: Optional[nn.Module]) -> bool:
        if model is None:
            return False
        # In PyTorch, we assume models output logits unless specified otherwise
        # Check if last layer has softmax
        last_module = list(model.modules())[-1]
        if isinstance(last_module, nn.Softmax):
            return False
        return True

    def _create_lr_schedule(self, optimizer: optim.Optimizer, epochs: int) -> Callable:
        """
        Create learning rate schedule with optional warmup.
        """
        schedule = (self.config.lr_schedule or 'cosine').lower()
        warmup_epochs = max(0, int(getattr(self.config, 'lr_warmup_epochs', 0)))
        total_epochs = max(1, epochs)
        initial_lr = float(self.config.initial_lr)
        min_lr_value = float(getattr(self.config, 'min_lr', 1e-7))

        def _warmup(epoch: int) -> float:
            if warmup_epochs <= 0:
                return initial_lr
            ratio = (epoch + 1) / float(warmup_epochs)
            return float(initial_lr * ratio)

        if schedule == 'cosine':
            min_factor = float(getattr(self.config, 'fine_tune_cosine_min_factor', 0.1))
            target_min_lr = max(min_lr_value, initial_lr * min_factor)
            self.logger.info(
                f"Training LR scheduler: cosine annealing (min_lr={target_min_lr:.2e}, warmup={warmup_epochs})"
            )
            # PyTorch CosineAnnealingLR doesn't support warmup directly, so use LambdaLR
            def cosine_schedule(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch) / initial_lr
                progress = (epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
                progress = np.clip(progress, 0.0, 1.0)
                cosine = 0.5 * (1 + np.cos(np.pi * progress))
                new_lr = target_min_lr + (initial_lr - target_min_lr) * cosine
                return float(max(min_lr_value, new_lr)) / initial_lr
            return LambdaLR(optimizer, lr_lambda=cosine_schedule)

        if schedule == 'exponential':
            decay_rate = float(self.config.lr_decay_rate)
            self.logger.info(
                f"Training LR scheduler: exponential decay (rate={decay_rate:.4f}, warmup={warmup_epochs})"
            )
            def exp_schedule(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch) / initial_lr
                effective_epoch = epoch - warmup_epochs
                new_lr = initial_lr * (decay_rate ** effective_epoch)
                return float(max(min_lr_value, new_lr)) / initial_lr
            return LambdaLR(optimizer, lr_lambda=exp_schedule)

        if schedule == 'step':
            decay_steps = max(1, int(self.config.lr_decay_steps))
            decay_rate = float(self.config.lr_decay_rate)
            self.logger.info(
                f"Training LR scheduler: step decay (steps={decay_steps}, rate={decay_rate:.3f}, warmup={warmup_epochs})"
            )
            def step_schedule(epoch: int) -> float:
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return _warmup(epoch) / initial_lr
                effective_epoch = epoch - warmup_epochs
                drops = max(0, effective_epoch // decay_steps)
                new_lr = initial_lr * (decay_rate ** drops)
                return float(max(min_lr_value, new_lr)) / initial_lr
            return LambdaLR(optimizer, lr_lambda=step_schedule)

        return None  # Constant LR

    def get_optimizer_and_loss(self,
                               model: nn.Module,
                               learning_rate: Optional[float] = None,
                               optimizer_name: Optional[str] = None) -> Tuple[optim.Optimizer, nn.Module]:
        """
        Get optimizer and loss for the model.
        """
        lr = learning_rate or self.config.initial_lr
        opt_name = optimizer_name or self.config.optimizer

        # Create optimizer
        if opt_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        elif opt_name.lower() == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.config.weight_decay)
        elif opt_name.lower() == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=self.config.momentum, nesterov=True)
        elif opt_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=lr, alpha=0.9, eps=1e-8)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        # Create loss function with label smoothing
        if self.config.label_smoothing > 0:
            # PyTorch doesn't have built-in label smoothing for CrossEntropy, need custom implementation
            class LabelSmoothingLoss(nn.Module):
                def __init__(self, smoothing=0.0, dim=-1):
                    super(LabelSmoothingLoss, self).__init__()
                    self.smoothing = smoothing
                    self.confidence = 1.0 - smoothing
                    self.dim = dim

                def forward(self, pred, target):
                    pred = pred.log_softmax(dim=self.dim)
                    with torch.no_grad():
                        true_dist = torch.zeros_like(pred)
                        true_dist.fill_(self.smoothing / (pred.size(self.dim) - 1))
                        if target.dim() > 1:
                            target = target.argmax(dim=1)
                        target = target.view(-1).long()
                        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
                    return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

            loss_fn = LabelSmoothingLoss(smoothing=self.config.label_smoothing)
        else:
            loss_fn = nn.CrossEntropyLoss()

        self.logger.info(f"Optimizer: {opt_name} with lr={lr}, label_smoothing={self.config.label_smoothing}")

        return optimizer, loss_fn

    def train_cifar(self,
                    model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    epochs: Optional[int] = None,
                    train_eval_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train the model with comprehensive anti-overfitting strategies
        """
        epochs = epochs or self.config.default_epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Initialize overfitting monitoring
        overfitting_monitor = OverfittingMonitor(
            patience=5,
            threshold=0.05
        )
        adaptive_reg = AdaptiveRegularization(
            initial_dropout=self.config.dropout_rate,
            max_dropout=min(0.7, self.config.dropout_rate * 2)
        )

        optimizer, loss_fn = self.get_optimizer_and_loss(model, self.config.initial_lr)

        # LR Scheduler
        scheduler = self._create_lr_schedule(optimizer, epochs) if self.config.lr_schedule else None

        # Early stopping
        early_stopping_patience = self.config.early_stopping_patience
        early_stop_counter = 0
        best_val_acc = 0.0

        # Reduce LR on plateau (PyTorch version)
        reduce_lr = ReduceLROnPlateau(optimizer, mode='min', factor=self.config.reduce_lr_factor,
                                      patience=self.config.reduce_lr_patience, min_lr=self.config.min_lr) if self.config.reduce_lr_patience > 0 else None

        # TensorBoard
        log_dir = os.path.join(self.config.tensorboard_dir, datetime.now().strftime("%d%m%Y-%H%M%S"))
        writer = SummaryWriter(log_dir)

        self.logger.info("Starting training with anti-overfitting strategies...")
        self.logger.info(f"Label smoothing: {self.config.label_smoothing}")
        self.logger.info(f"Dropout rate: {self.config.dropout_rate}")
        self.logger.info(f"L2 regularization: {self.config.l2_regularization}")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            total = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / total

            # Validation
            val_loss, val_acc = self._evaluate(model, val_loader, loss_fn, device)

            # Log to TensorBoard
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/acc', val_acc, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])

            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_model(model, f'best_model_{epoch:02d}_{val_acc:.3f}.pth')
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Early stopping
            if early_stopping_patience > 0 and early_stop_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            # Step scheduler
            if scheduler:
                scheduler.step()
            if reduce_lr:
                reduce_lr.step(val_loss)

            # Clean train evaluation if provided
            if train_eval_loader is not None:
                clean_loss, clean_acc = self._evaluate(model, train_eval_loader, loss_fn, device)
                writer.add_scalar('clean/loss', clean_loss, epoch)
                writer.add_scalar('clean/acc', clean_acc, epoch)
                self.logger.info(f"Epoch {epoch + 1}: clean_train_loss={clean_loss:.4f}, clean_train_accuracy={clean_acc:.4f}")

        writer.close()

        # Save final model and history
        self._save_model(model, 'final_model.pth')
        self._save_history()

        # Plot overfitting analysis
        overfitting_monitor.plot_overfitting_analysis(
            save_path=os.path.join(self.config.plots_dir, 'overfitting_analysis.png')
        )

        # Log final results
        final_train_acc = self.history['train_acc'][-1]
        final_val_acc = self.history['val_acc'][-1]
        overfitting_score = overfitting_monitor.get_overfitting_score()
        self.logger.info(f"Training completed:")
        self.logger.info(f" Final training accuracy: {final_train_acc:.4f}")
        self.logger.info(f" Final validation accuracy: {final_val_acc:.4f}")
        self.logger.info(f" Final overfitting score: {overfitting_score:.3f}")

        if overfitting_score > 0.5:
            self.logger.warning("Model shows signs of overfitting. Consider:")
            self.logger.warning(" - Increasing data augmentation")
            self.logger.warning(" - Increasing dropout/regularization")
            self.logger.warning(" - Reducing model capacity")
            self.logger.warning(" - Gathering more training data")

        return self.history

    def _evaluate(self, model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss += loss_fn(outputs, batch_y).item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        return loss / len(loader), correct / total

    def _save_model(self, model: nn.Module, filename: str):
        """Save model to file"""
        filepath = os.path.join(self.config.models_dir, filename)
        torch.save(model.state_dict(), filepath)
        self.logger.info(f"Model saved to {filepath}")

    def _save_history(self):
        """Save training history to JSON"""
        filepath = os.path.join(self.config.results_dir, 'training_history.json')
        save_dict = {key: [float(v) for v in values] for key, values in self.history.items()}
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=4)
        self.logger.info(f"Training history saved to {filepath}")

    def load_model(self, model: nn.Module, filepath: str) -> nn.Module:
        """Load model from file"""
        model.load_state_dict(torch.load(filepath))
        self.logger.info(f"Model loaded from {filepath}")
        return model

    def fine_tune_cifar(self,
                        model: nn.Module,
                        train_loader: DataLoader,
                        val_loader: DataLoader,
                        epochs: int = 10,
                        learning_rate: Optional[float] = None,
                        log_dir_suffix: Optional[str] = None,
                        stage_log_root: Optional[str] = None,
                        stage_index: Optional[int] = None,
                        train_eval_loader: Optional[DataLoader] = None,
                        teacher_model: Optional[nn.Module] = None,
                        kd_alpha: Optional[float] = None,
                        kd_temperature: Optional[float] = None,
                        lr_schedule: Optional[str] = None,
                        frequency_regularizer_config: Optional[Dict[str, object]] = None) -> Tuple[nn.Module, Dict[str, object]]:
        """
        Fine-tune model on a single dataset after pruning
        """
        self.logger.info("Starting fine-tuning on single dataset...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if teacher_model:
            teacher_model.to(device)
            teacher_model.eval()

        lr = learning_rate if learning_rate is not None else self.config.pruned_growth_lr
        self.logger.info(f"Fine-tune learning rate set to {lr:.2e}")

        optimizer, loss_fn = self.get_optimizer_and_loss(model, lr, 'adamw')  # Assuming AdamW for fine-tune

        # Setup logging
        if stage_index is not None:
            run_identifier = f"stage{int(stage_index):02d}"
            root = stage_log_root or os.path.join(self.config.tensorboard_dir, "regrow_runs", "stages_shared")
            os.makedirs(root, exist_ok=True)
            train_log_dir = os.path.join(root, f"stage{stage_index}-train")
            val_log_dir = os.path.join(root, f"stage{stage_index}-validation")
        else:
            run_stamp = datetime.now().strftime("%d%m%Y-%H%M%S")
            suffix = (log_dir_suffix or "fine_tune").replace(" ", "_")
            run_identifier = f"{run_stamp}_{suffix}"
            root = os.path.join(
                self.config.tensorboard_dir,
                "regrow_runs",
                f"{run_stamp}_{suffix}"
            )
            train_log_dir = os.path.join(root, "train")
            val_log_dir = os.path.join(root, "validation")
        os.makedirs(train_log_dir, exist_ok=True)
        os.makedirs(val_log_dir, exist_ok=True)
        split_writer = SplitTensorBoard(train_log_dir, val_log_dir)

        checkpoint_dir = os.path.join(self.config.models_dir, "fine_tune_checkpoints", run_identifier)
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{run_identifier}_best.pth")

        # LR Schedule
        lr_schedule_mode = (lr_schedule or getattr(self.config, 'fine_tune_lr_schedule', 'plateau')).lower()
        if lr_schedule_mode == 'exponential':
            decay_rate = float(getattr(self.config, 'fine_tune_exp_decay', 0.96))
            scheduler = ExponentialLR(optimizer, gamma=decay_rate)
        elif lr_schedule_mode == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=self.config.min_lr)
        elif lr_schedule_mode == 'step':
            step_epochs = int(getattr(self.config, 'fine_tune_step_decay_epochs', 10))
            step_rate = float(getattr(self.config, 'fine_tune_step_decay_rate', 0.5))
            scheduler = StepLR(optimizer, step_size=step_epochs, gamma=step_rate)
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=self.config.reduce_lr_factor,
                                          patience=max(3, self.config.reduce_lr_patience // 2),
                                          min_lr=self.config.min_lr)

        # Distillation or Frequency Reg
        use_kd = teacher_model is not None
        freq_model = None
        use_freq_reg = False
        if frequency_regularizer_config:
            # Assuming PyTorch implementation of FrequencyRegularizedModel and SpectralEntropyRegularizer
            layer_names = frequency_regularizer_config.get('layer_names') or []
            targets = frequency_regularizer_config.get('targets') or {}
            beta = float(frequency_regularizer_config.get('beta', 0.01))
            if layer_names and targets:
                spectral_regularizer = SpectralEntropyRegularizer(
                    model=model,
                    layer_names=layer_names,
                    target_entropies=targets,
                    beta=beta,
                    layer_weights=frequency_regularizer_config.get('layer_weights'),
                )
                freq_model = FrequencyRegularizedModel(model, spectral_regularizer)
                use_freq_reg = True
                use_kd = False
                self.logger.info(
                    f"Frequency regularization enabled on {len(layer_names)} layer(s) "
                    f"(beta={beta:.3f})."
                )

        if use_kd:
            alpha = kd_alpha if kd_alpha is not None else self.config.distill_alpha
            temperature = kd_temperature if kd_temperature is not None else self.config.distill_temperature
            self.logger.info(
                f"Knowledge distillation enabled (alpha={alpha:.3f}, temperature={temperature:.2f})"
            )
            # Assuming PyTorch Distiller implementation
            distiller = Distiller(
                student=model,
                teacher=teacher_model,
                alpha=alpha,
                temperature=temperature,
            )
            student_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) if self.config.label_smoothing > 0 else nn.CrossEntropyLoss()
            distill_loss_fn = nn.KLDivLoss(reduction='batchmean')

        best_val_acc = 0.0
        early_stop_counter = 0
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            if use_kd:
                train_loss, train_acc = self._train_epoch_distill(distiller, train_loader, optimizer, student_loss_fn, distill_loss_fn, device, alpha, temperature)
            elif use_freq_reg:
                train_loss, train_acc = self._train_epoch(model=freq_model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)
            else:
                train_loss, train_acc = self._train_epoch(model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn, device=device)

            val_loss, val_acc = self._evaluate(model, val_loader, loss_fn, device)

            logs = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            split_writer.write_epoch(epoch, logs)

            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), checkpoint_path)
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= 7:  # Hardcoded patience
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            # Step scheduler
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()

        # Load best checkpoint
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            self.logger.info(f"Loaded best fine-tune checkpoint from {checkpoint_path}")

        best_model_filename = f"{run_identifier}_best_model.pth"
        best_model_path = os.path.join(self.config.models_dir, best_model_filename)
        self._save_model(model, best_model_filename)

        self.logger.info(f"Fine-tuning completed. Final accuracy: {history['accuracy'][-1]:.4f}")

        clean_metrics = None
        if train_eval_loader is not None:
            clean_loss, clean_acc = self._evaluate(model, train_eval_loader, loss_fn, device)
            clean_metrics = {'loss': clean_loss, 'accuracy': clean_acc}
            self.logger.info(f"Clean training-set evaluation after fine-tune -> loss: {clean_loss:.4f}, accuracy: {clean_acc:.4f}")

        # Best val metrics
        best_val_metrics = {}
        val_acc_hist = history.get('val_accuracy')
        if val_acc_hist:
            best_idx = int(np.argmax(val_acc_hist))
            best_val_metrics['val_accuracy'] = float(val_acc_hist[best_idx])
            if 'val_loss' in history:
                best_val_metrics['val_loss'] = float(history['val_loss'][best_idx])

        if os.path.isdir(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)

        metadata = {
            'history': history,
            'epochs_ran': len(history.get('loss', [])),
            'log_dirs': {
                'train': train_log_dir,
                'validation': val_log_dir
            },
            'clean_train_metrics': clean_metrics,
            'best_val_metrics': best_val_metrics if best_val_metrics else None,
            'checkpoint_path': checkpoint_path if os.path.exists(checkpoint_path) else None,
            'best_model_path': best_model_path,
        }

        split_writer.close()

        if use_kd:
            del distiller
            gc.collect()
        elif use_freq_reg:
            del freq_model
            gc.collect()

        return model, metadata

    def _train_epoch(self, model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, device: torch.device) -> Tuple[float, float]:
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        return train_loss / len(loader), train_correct / total

    def _train_epoch_distill(self, distiller, loader: DataLoader, optimizer: optim.Optimizer, student_loss_fn: nn.Module, distill_loss_fn: nn.Module, device: torch.device, alpha: float, temperature: float) -> Tuple[float, float]:
        # Implement distillation training loop here, similar to above but using distiller
        # Assuming distiller has a forward that computes combined loss
        train_loss = 0.0
        train_correct = 0
        total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            # Assume distiller computes student_outputs, teacher_outputs, etc.
            combined_loss = distiller.compute_loss(batch_x, batch_y)  # Need to define this in Distiller
            combined_loss.backward()
            optimizer.step()
            train_loss += combined_loss.item()
            student_outputs = distiller.student(batch_x)  # Assume
            _, predicted = student_outputs.max(1)
            total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        return train_loss / len(loader), train_correct / total

    def simple_finetune(
        self,
        model_or_path: Union[str, nn.Module],
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        teacher_model: Optional[nn.Module] = None,
        train_eval_loader: Optional[DataLoader] = None,
        log_dir_suffix: Optional[str] = None,
        frequency_regularizer_config: Optional[Dict[str, object]] = None,
        kd_alpha: Optional[float] = None,
        kd_temperature: Optional[float] = None,
    ) -> Tuple[nn.Module, Dict[str, object]]:
        """Load a checkpoint (or reuse an in-memory model) and run extra fine-tuning.
        """
        if isinstance(model_or_path, (str, bytes, os.PathLike)):
            model = self.load_model(model_or_path)  # Note: Need to provide model instance to load_state_dict
            # Assuming user provides model class instance, load here would require model = ModelClass(); model = self.load_model(model, path)
        else:
            model = model_or_path

        tuned_model, metadata = self.fine_tune_cifar(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs or self.config.hybrid_finetune_epochs,
            learning_rate=learning_rate or self.config.pruned_growth_lr,
            log_dir_suffix=log_dir_suffix or "manual_finetune",
            train_eval_loader=train_eval_loader,
            teacher_model=teacher_model,
            kd_alpha=kd_alpha,
            kd_temperature=kd_temperature,
            frequency_regularizer_config=frequency_regularizer_config,
        )
        return tuned_model, metadata