#!/usr/bin/env python3
"""Main training script for vital signs monitoring system."""

import argparse
import sys
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.train import run_experiment
from src.utils import set_seed, setup_logging, validate_config
from omegaconf import DictConfig, OmegaConf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train vital signs monitoring models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    
    # Override config with command line arguments
    if args.output_dir:
        config.output.save_dir = args.output_dir
    if args.seed:
        config.seed = args.seed
    if args.log_level:
        config.logging.level = args.log_level
    
    # Validate configuration
    config = validate_config(config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.save_dir) / f"experiment_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "training.log"
    logger = setup_logging(
        level=config.logging.level,
        log_file=str(log_file)
    )
    
    logger.info("Starting vital signs monitoring training")
    logger.info(f"Configuration: {config}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Run experiment
        results = run_experiment(
            config=OmegaConf.to_container(config, resolve=True),
            output_dir=output_dir,
            random_seed=config.seed
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Results: {results}")
        
        # Save configuration
        config_save_path = output_dir / "config.yaml"
        with open(config_save_path, 'w') as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Configuration saved to {config_save_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
