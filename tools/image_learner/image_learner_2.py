import yaml
import pandas as pd
import warnings

def generate_ludwig_config(raw_data=None, encoder_type="resnet50", sample_name_col=None, split_col=None, dataset_path=None):
    # Base config structure
    config = {
        "output_features": [
            {"name": "label", "type": "category"}
        ],
        "trainer": {
            "epochs": 10,
            "batch_size": 32
        }
    }

    # Handle input feature based on raw_data presence
    if raw_data:  # Image zip is provided
        config["input_features"] = [
            {
                "name": "image_path",
                "type": "image",
                "encoder": encoder_type,
                "preprocessing": {
                    "height": 224,
                    "width": 224
                }
            }
        ]
        config["input_features"][0]["preprocessing"]["in_memory"] = False
    else:  # No image zip, use vector embeddings
        # Load dataset to check for 'embeddings' column
        df = pd.read_csv(dataset_path)  # Assuming CSV for simplicity; adjust as needed
        if "embeddings" not in df.columns:
            raise ValueError("Dataset must contain 'embeddings' column when image zip is absent.")
        config["input_features"] = [
            {
                "name": "embeddings",  # Column storing the vectors aggregated as string white-separated
                "type": "vector"
            }
        ]

    # Handle split logic based on sample_name_col and split_col
    if split_col:  # If split column exists
        config["split"] = {
            "type": "fixed",
            "column": split_col
        }
    else:
        # Load dataset to check sample_name_col (simulated here)
        # In practice, this would be a real DataFrame from input
        df = pd.DataFrame({"sample_name": ["a", "b", "a"] if sample_name_col else []})
        if sample_name_col and df["sample_name"].duplicated().any():
            warnings.warn("Duplicate sample names detected; using random split.")
        config["split"] = {
            "type": "random",
            "split_probabilities": [0.7, 0.15, 0.15]  # train, val, test
        }

    # Output the config as YAML
    return yaml.dump(config, default_flow_style=False)


# Example usage
config_yaml = generate_ludwig_config(
    raw_data=None,  # No image zip
    encoder_type="resnet50",
    sample_name_col="sample_name",
    split_col=None,
    dataset_path="path/to/dataset.csv"
)
print(config_yaml)
