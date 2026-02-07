import os
import argparse
import torch
from config import Config
from utils.logger import Logger
from data.data_loader import HSGSP_DataLoader
from models.vgg16 import VGG16
from training.trainer import HSGSPTrainer
from training.evaluator import ModelEvaluator
from utils.visualization import Visualizer
from pruning.hybrid_baseline import HybridFrequencyBaseline

def _load_datasets(data_loader: HSGSP_DataLoader, task: str, logger: Logger):
    if task == 'cifar10':
        logger.info('Loading CIFAR-10 dataset...')
        train_loader, val_loader, test_loader, train_clean_loader = data_loader.load_cifar10()
        return 'CIFAR-10', train_loader, val_loader, test_loader, train_clean_loader
    if task == 'cifar100':
        logger.info('Loading CIFAR-100 dataset...')
        train_loader, val_loader, test_loader, train_clean_loader = data_loader.load_cifar100()
        return 'CIFAR-100', train_loader, val_loader, test_loader, train_clean_loader
    raise ValueError(f"Unsupported task: {task}")

def _build_model(config: Config, task: str):
    builder = VGG16(config)
    if task == 'cifar10':
        return builder.build_vgg16_model(
            num_classes=config.num_classes_cifar10,
            input_shape=config.input_shape_cifar10,
        )
    if task == 'cifar100':
        return builder.build_vgg16_model(
            num_classes=config.num_classes_cifar100,
            input_shape=config.input_shape_cifar100,
        )
    raise ValueError(f"Unsupported task for VGG16: {task}")

def main(args):
    config = Config(task=args.task)
    if args.batch_size:
        config.batch_size = args.batch_size
    logger = Logger(config)
    logger.info('Starting hybrid pruning experiment')
    data_loader = HSGSP_DataLoader(config)
    trainer = HSGSPTrainer(config)
    evaluator = ModelEvaluator(config)
    visualizer = Visualizer(config)
    hybrid_baseline = HybridFrequencyBaseline(config, trainer, evaluator)
    dataset_name, train_loader, val_loader, test_loader, train_clean_loader = _load_datasets(data_loader, args.task, logger)
    pruned_model = None
    if args.pruned_model_path:
        logger.info(f"Loading pruned model from {args.pruned_model_path}")
        pruned_model = _build_model(config, args.task)  # Instantiate first
        pruned_model.load_state_dict(torch.load(args.pruned_model_path))
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")
        model = _build_model(config, args.task)  # Instantiate first
        model.load_state_dict(torch.load(args.model_path))
    else:
        logger.info('Building new model...')
        model = _build_model(config, args.task)
    logger.info('Model summary:')
    # For PyTorch, use a summary function or print(model)
    print(model)  # Simple print for now
    if args.train:
        logger.info('Starting training phase...')
        history = trainer.train_cifar(
            model,
            train_loader,
            val_loader,
            epochs=args.epochs or config.default_epochs,
            train_eval_loader=train_clean_loader,
        )
        model_save_path = os.path.join(config.models_dir, f"{args.task}_trained_model.pth")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Trained model saved at {model_save_path}")
        visualizer.plot_training_history(
            history,
            save_path=os.path.join(config.plots_dir, 'training_history.png'),
        )
    baseline_for_eval = None
    if args.eval:
        baseline_for_eval = _build_model(config, args.task)
        baseline_for_eval.load_state_dict(model.state_dict())
    hybrid_history = None
    if args.prune:
        logger.info('Running hybrid pruning baseline...')
        activation_source = train_clean_loader or train_loader or val_loader
        pruned_model, hybrid_history = hybrid_baseline.run_pipeline(
            model=model,
            train_dl=train_loader,
            val_dl=val_loader,
            train_eval_dl=train_clean_loader,
            activation_dl=activation_source,
        )
        model = pruned_model
        for record in hybrid_history or []:
            metrics = record.get('metrics', {})
            loss_val = metrics.get('loss')
            acc_val = metrics.get('accuracy')
            kappa = record.get('kappa_ratio', 0.0)
            loss_str = f"{loss_val:.4f}" if loss_val is not None else 'nan'
            acc_str = f"{acc_val:.4f}" if acc_val is not None else 'nan'
            logger.info(
                f"Hybrid iteration {record.get('iteration')}: "
                f"kappa_ratio={kappa:.3f}, val_loss={loss_str}, val_acc={acc_str}"
            )
        pruned_model_path = os.path.join(config.models_dir, f"{args.task}_hybrid_pruned.pth")
        torch.save(model.state_dict(), pruned_model_path)
        logger.info(f"Hybrid pruned model saved at {pruned_model_path}")
    if args.simple_finetune:
        logger.info('Running manual simple fine-tune...')
        tuned_model, finetune_meta = trainer.fine_tune_cifar(
            model=pruned_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.simple_finetune_epochs,
            learning_rate=config.simple_finetune_lr,
            log_dir_suffix="manual_finetune",
            train_eval_loader=train_clean_loader,
            # teacher_model=model if bool(model) else None,
            # kd_alpha=config.distill_alpha,
            # kd_temperature=config.distill_temperature
        )
        model = tuned_model
        pruned_model = tuned_model
        best_path = finetune_meta.get('best_model_path') if isinstance(finetune_meta, dict) else None
        if best_path:
            logger.info(f"Simple fine-tune best model saved at {best_path}")
    if args.eval:
        logger.info('Starting evaluation phase...')
        original_model = baseline_for_eval or model
        logger.info(f"Evaluating original model on {dataset_name}...")
        original_results = evaluator.evaluate_model(original_model, test_loader, dataset_name)
        logger.info('=' * 50)
        logger.info(f"Original Model Results on {dataset_name}:")
        logger.info(f" Parameters: {original_results['total_params']:,}")
        logger.info(f" Model size: {original_results['model_size_mb']:.2f} MB")
        logger.info(f" FLOPs: {original_results['flops'] / 1e6:.2f} MFLOPs")
        logger.info(f" Inference time: {original_results['inference_time_ms']:.2f} ms")
        logger.info(f" Loss: {original_results['loss']:.2f}")
        logger.info(f" Accuracy: {original_results['accuracy']:.4f}")
        logger.info(f" Top-5 Accuracy: {original_results['top5_accuracy']:.2f}")
        logger.info('=' * 50)
        if pruned_model is not None:
            logger.info(f"Evaluating pruned model on {dataset_name}...")
            pruned_results = evaluator.evaluate_model(pruned_model, test_loader, dataset_name)
            logger.info('=' * 50)
            logger.info(f"Pruned Model Results on {dataset_name}:")
            logger.info(f" Parameters: {pruned_results['total_params']:,}")
            logger.info(f" Model size: {pruned_results['model_size_mb']:.2f} MB")
            logger.info(f" FLOPs: {pruned_results['flops'] / 1e6:.2f} MFLOPs")
            logger.info(f" Inference time: {pruned_results['inference_time_ms']:.2f} ms")
            logger.info(f" Loss: {pruned_results['loss']:.2f}")
            logger.info(f" Accuracy: {pruned_results['accuracy']:.4f}")
            logger.info(f" Top-5 Accuracy: {pruned_results['top5_accuracy']:.2f}")
            logger.info('=' * 50)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hybrid Frequency-Aware Pruning for CIFAR models')
    parser.add_argument('--task', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--prune', action='store_true', help='Run hybrid pruning')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model(s)')
    parser.add_argument('--model_path', type=str, default=None, help='Path to pre-trained model')
    parser.add_argument('--pruned_model_path', type=str, default=None, help='Path to pre-trained pruned model')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size override')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
    parser.add_argument('--simple_finetune', action='store_true', help='Simple finetune the model(s)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    main(args)