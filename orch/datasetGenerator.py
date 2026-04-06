from pathlib import Path

import datasets
import h5py
import huggingface_hub as hf
import numpy as np
from datasets import Dataset, Features, Sequence, Value

# ============================================================================
# Configuration — edit these variables instead of using CLI arguments
# ============================================================================

INPUT_PATH = "./"                       # Path to a single .hdf5 file or directory containing .hdf5 files
DATASET_NAME = "Key-Generation"         # Dataset repository name on Hugging Face (without owner)
REPO_NAME = "CAAI-FAU"                  # Dataset owner/org. Use "" to use your logged-in username
CONFIG_PREFIX = "WiFi-Powder"           # Prefix for config names; final config is <prefix>-<environment>-Nodes-<nodeIds>
PRIVATE = True                          # Set to False to push as public dataset

# ============================================================================

OFDM_REQUIRED_KEYS = {
    "iq_I",
    "iq_Q",
    "csi_I",
    "csi_Q",
    "chan_est_samples_I",
    "chan_est_samples_Q",
    "ids",
    "rx",
    "tx",
    "channel",
    "instance",
}


class DatasetGenerator:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data_dict = self.generate_dict_from_hdf5()
        self.hf_dataset = self._to_hf_dataset(self.data_dict)

    def save_to_huggingface(self, dataset_name, config_name, repo_name="CAAI-FAU", private=True):
        try:
            username = hf.whoami()["name"]
            print(f"Logged in as: {username}")
        except Exception:
            print(
                "Could not verify Hugging Face login. "
                "Make sure you ran `huggingface-cli login`."
            )
            raise

        owner = username if repo_name == "" else repo_name
        repo_id = f"{owner}/{dataset_name}"
        print(f"Pushing dataset to Hugging Face Hub: {repo_id} (config: {config_name})")
        self.hf_dataset.push_to_hub(
            repo_id=repo_id,
            private=private,
            config_name=config_name,
        )
        print("Dataset pushed to Hugging Face Hub successfully.")

    def read_hdf5_file(self):
        data = {}
        with h5py.File(self.file_name, "r") as data_file:
            keys = list(data_file.keys())
            print("Keys in HDF5 file:", keys)
            for key in keys:
                data[key] = data_file[key][:]
        return data

    def generate_dict_from_hdf5(self):
        raw_data = self.read_hdf5_file()
        normalized = {}
        for key, value in raw_data.items():
            normalized[key] = self._normalize_hdf5_column(value)
        self._validate_columns(normalized)
        return normalized

    def _normalize_hdf5_column(self, value):
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == 1 and value.dtype.kind != "S":
                value = value[0]
            if isinstance(value, np.ndarray):
                return self._array_to_python(value)
        return value

    def _array_to_python(self, arr):
        if arr.dtype.kind == "S":
            return [x.decode("utf-8") for x in arr.tolist()]
        if arr.dtype.kind in ("i", "u"):
            if arr.ndim == 1:
                return [int(x) for x in arr.tolist()]
            return [[int(x) for x in row] for row in arr.tolist()]
        if arr.dtype.kind == "f":
            if arr.ndim == 1:
                return [float(x) for x in arr.tolist()]
            return [[float(x) for x in row] for row in arr.tolist()]
        return arr.tolist()

    def _validate_columns(self, data_dict):
        if not OFDM_REQUIRED_KEYS.issubset(set(data_dict.keys())):
            missing = sorted(list(OFDM_REQUIRED_KEYS - set(data_dict.keys())))
            raise ValueError(
                f"File {self.file_name} is missing required OFDM keys: {missing}"
            )

        lengths = {key: len(value) for key, value in data_dict.items() if hasattr(value, "__len__")}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            raise ValueError(
                f"Column length mismatch in {self.file_name}: {lengths}"
            )

    def _to_hf_dataset(self, data_dict):
        feature_spec = Features(
            {
                "iq_I": Sequence(Value("float32")),
                "iq_Q": Sequence(Value("float32")),
                "pilots_I": Sequence(Value("float32")) if "pilots_I" in data_dict else Sequence(Value("float32")),
                "pilots_Q": Sequence(Value("float32")) if "pilots_Q" in data_dict else Sequence(Value("float32")),
                "csi_I": Sequence(Value("float32")),
                "csi_Q": Sequence(Value("float32")),
                "chan_est_samples_I": Sequence(Value("float32")),
                "chan_est_samples_Q": Sequence(Value("float32")),
                "ids": Value("int32"),
                "rx": Value("int32"),
                "tx": Value("int32"),
                "channel": Value("int32"),
                "instance": Value("int32"),
                "timestamp": Value("float64") if "timestamp" in data_dict else Value("float64"),
            }
        )

        # Ensure optional fields always exist so a single schema works for all files.
        if "pilots_I" not in data_dict:
            data_dict["pilots_I"] = [[] for _ in range(len(data_dict["ids"]))]
        if "pilots_Q" not in data_dict:
            data_dict["pilots_Q"] = [[] for _ in range(len(data_dict["ids"]))]
        if "timestamp" not in data_dict:
            data_dict["timestamp"] = [0.0 for _ in range(len(data_dict["ids"]))]

        return Dataset.from_dict(data_dict, features=feature_spec)


def collect_hdf5_files(input_path):
    path = Path(input_path)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    return sorted(path.glob("*.hdf5"))


def push_files_to_hub(files, dataset_name, repo_name, config_prefix, private=True):
    if not files:
        raise ValueError("No .hdf5 files found to push.")

    for file_path in files:
        
        print(f"\nProcessing file: {file_path}")
        generator = DatasetGenerator(str(file_path))
        # Split the file_path by _ to get the dataset info
        environment = file_path.stem.split("_")[1]
        if environment == "OTALab":
            environment = "OTA-Lab"
        elif environment == "OTADense":
            environment = "OTA-Dense"
        probeType = file_path.stem.split("_")[3]
        # numProbes = file_path.stem.split("_")[4]
        nodeIds = file_path.stem.split("_")[5]
        # timestamp = file_path.stem.split("_")[6]
        
        config_name = f"{config_prefix}-{environment}-Nodes-{nodeIds}" if config_prefix else f"{probeType}-{environment}-Nodes-{nodeIds}"
        print(f"Config name: {config_name}")
        generator.save_to_huggingface(
            dataset_name=dataset_name,
            config_name=config_name,
            repo_name=repo_name,
            private=private,
        )
        loaded = datasets.load_dataset(f"{repo_name}/{dataset_name}", config_name)
        print("Verified upload. Loaded dataset:", loaded)


if __name__ == "__main__":
    files = collect_hdf5_files(INPUT_PATH)
    target_repo = REPO_NAME if REPO_NAME != "" else hf.whoami()["name"]
    push_files_to_hub(
        files=files,
        dataset_name=DATASET_NAME,
        repo_name=target_repo,
        config_prefix=CONFIG_PREFIX,
        private=PRIVATE,
    )