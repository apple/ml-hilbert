#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import os
import json
from datetime import datetime
import hydra
from omegaconf import DictConfig
from src.experiment.run_formal_to_formal import run_formal_to_formal
from src.experiment.run_async_hilbert import run_async_hilbert
from src.tools.directories import make_dirs, write_string_to_file
import logging
logger = logging.getLogger(__name__)

def run_experiment(cfg):
    if cfg.experiment.name == 'formal_to_formal':
        results = run_formal_to_formal(cfg)
    elif cfg.experiment.name == 'async_hilbert':
        results = run_async_hilbert(cfg)    
    else:
        raise NotImplementedError
    # Save the results
    make_dirs(os.path.join(cfg.results_dir, cfg.experiment.name))
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{cfg.experiment.name}_{cfg.data.name}_results_{timestamp}.json"
    filepath = os.path.join(cfg.results_dir, cfg.experiment.name, filename)
    
    # Save results as JSON
    results_str = json.dumps(results, indent=2)
    write_string_to_file(results_str, filepath)

    logger.info("Results saved to: %s", filepath)


@hydra.main(version_base=None, config_path="../configs", config_name="run")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration management."""
    
    # run the correct experiment based on the config
    run_experiment(cfg)

if __name__ == "__main__":
    main()