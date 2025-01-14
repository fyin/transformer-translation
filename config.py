from pathlib import Path

def get_config():
    """
    Get the configuration for the transformer model.
    """
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "embedding_dim": 512,
        "datasource": "Helsinki-NLP/opus_books",
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    """
    Get the path to the weights file for a specific epoch.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return Path(model_folder) / model_filename

def latest_weights_file_path(config):
    """
    Get the path to the latest weights file.
    """
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return weights_files[-1]