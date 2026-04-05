import math
import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from deep_learning_models import identity_loss, TripletNet_Channel, RNN_TripletNet_Channel, ResNet_QuadrupletNet_Channel, FeedForward_QuadrupletNet_Channel, RNN_QuadrupletNet_Channel, RNN_QuadrupletNet_Channel_Simple, Transformer_QuadrupletNet_Channel, AE_QuadrupletNet_Channel, ResNet_HashNet_Channel, RNN_HashNet_Channel
from DatasetHandler import DatasetHandler, ChannelSpectrogram, ChannelIQ, ChannelPolar, load_cached_dataset, SOURCE_OTA_LAB, SOURCE_OTA_DENSE, SOURCE_SIONNA
from dataset_visualization import plot_data_scenario, plot_avg_min_max_power_scenario
from SionnaDataGenerator import generate_multi_scenario_data

import time
from TestChannelFingerprinting import test_model
import h5py
from collections import Counter
import numpy as np
import os
import random
import contextlib
from typing import Any, Dict, List
# Optional: force CPU when CUDA/cuBLAS is misconfigured.
# Set POWDER_FORCE_CPU=1 in the environment to disable GPU.
if os.environ.get("POWDER_FORCE_CPU") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import matplotlib.pyplot as plt

DEFAULT_HOME_DIR = "/home/Research/POWDER/"
DEFAULT_VENV_PYTHON = "/home/Research/.venvs/siwn-tf216/bin/python"

def configure_visible_gpu_memory_growth():
    """Enable memory growth on whichever GPUs are currently visible."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as exc:
        print(exc)

configure_visible_gpu_memory_growth()

def training_device_context():
    """Use CPU only when explicitly requested via POWDER_FORCE_CPU=1."""
    if os.environ.get("POWDER_FORCE_CPU") == "1":
        return tf.device("/CPU:0")
    return contextlib.nullcontext()

def save_feature_extractor_keras(model, keras_output_path: str):
    """Save a Keras feature extractor using native .keras format and validate reload."""
    model.save(keras_output_path)
    try:
        tf.keras.models.load_model(keras_output_path, compile=False, safe_mode=False)
    except TypeError:
        # Older tf.keras may not support safe_mode kwarg.
        tf.keras.models.load_model(keras_output_path, compile=False)

def set_reproducible(seed: int, deterministic: bool = True, max_threads: int = 1):
    """Set seeds and deterministic execution for reproducibility.

    Note: Some ops may remain nondeterministic on certain GPUs/versions.
    """
    # Python, NumPy, TensorFlow seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        # Sets TF, NumPy, and Python seeds together (TF 2.9+)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        tf.random.set_seed(seed)

    # Deterministic behavior (TF 2.12+)
    if deterministic:
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        try:
            tf.config.experimental.enable_op_determinism(True)
        except Exception:
            pass

    # Constrain parallelism to reduce nondeterminism from threads
    try:
        tf.config.threading.set_intra_op_parallelism_threads(max_threads)
        tf.config.threading.set_inter_op_parallelism_threads(max_threads)
    except Exception:
        pass

def get_default_sionna_config(signal_type: str = "Sinusoid") -> Dict[str, Any]:
    return {
        "signal_type": signal_type,
        "num_probes": 100,
        "num_samples": 8192,
        "num_bins": 1024,
        "backend": "gnuradio",
        "noise_voltage": 0.01,
        "seed": 42,
        "scenarios": [
            ["EBC", "Guest House", "Moran"],
            ["EBC", "Guest House", "USTAR"],
            ["EBC", "Moran", "Guest House"],
            ["EBC", "Moran", "USTAR"],
            ["EBC", "USTAR", "Guest House"],
            ["EBC", "USTAR", "Moran"],
            ["Guest House", "Moran", "EBC"],
            ["Guest House", "Moran", "USTAR"],
            ["Guest House", "USTAR", "Moran"],
            ["Guest House", "USTAR", "EBC"],
            ["USTAR", "Moran", "EBC"],
            ["USTAR", "Moran", "Guest House"],
        ],
    }

def get_default_signal_node_ids() -> Dict[str, Dict[str, List[List[int]]]]:
    return {
        "Sinusoid": {
            "OTA-Lab": [
                [1, 2, 3],
                [1, 4, 5],
                [1, 4, 8],
                [2, 4, 3],
                [4, 2, 5],
                [4, 2, 8],
                [4, 8, 5],
                [5, 7, 8],
                [5, 8, 7],
                [8, 4, 1],
                [8, 5, 1],
            ],
            "OTA-Dense": [],
        },
        "PN-Sequence": {
            "OTA-Lab": [],
            "OTA-Dense": [],
        },
        "deltaPulse": {
            "OTA-Lab": [
                [5, 8, 2],
                [5, 8, 3],
                [5, 8, 6],
                [5, 8, 7],
                [6, 7, 2],
                [6, 7, 3],
                [6, 7, 4],
                [6, 7, 5],
                [6, 7, 8],
            ],
            "OTA-Dense": [],
        },
        "WiFi": {
            "OTA-Lab": [
                [5, 6, 2],
                [5, 6, 3],
                [5, 6, 4],
                [5, 6, 7],
                [5, 6, 8],
                [5, 7, 2],
                [5, 7, 3],
                [5, 7, 4],
                [5, 7, 6],
                [5, 7, 8],
                [5, 8, 2],
                [5, 8, 3],
                [5, 8, 6],
                [5, 8, 7],
                [6, 7, 2],
            ],
            "OTA-Dense": [],
        },
    }

def build_default_training_node_configurations(signal_type: str = "Sinusoid") -> Dict[str, Dict[str, Any]]:
    node_ids = get_default_signal_node_ids()
    if signal_type not in node_ids:
        raise ValueError(f"Unsupported signal_type={signal_type!r}")
    return {
        "OTA-Lab": {
            "dataset_name": "Key-Generation",
            "config_name": f"{signal_type}-Powder-OTA-Lab-Nodes",
            "repo_name": "CAAI-FAU",
            "node_Ids": copy.deepcopy(node_ids[signal_type]["OTA-Lab"]),
        },
        "OTA-Dense": {
            "dataset_name": "Key-Generation",
            "config_name": f"{signal_type}-Powder-OTA-Dense-Nodes",
            "repo_name": "CAAI-FAU",
            "node_Ids": copy.deepcopy(node_ids[signal_type]["OTA-Dense"]),
        },
    }

def resolve_model_type(experiment_spec: Dict[str, Any]) -> str:
    family = experiment_spec["family"]
    return "QuadrupletNet" if family == "RNN-Simple" else family

def resolve_network_type(experiment_spec: Dict[str, Any]) -> str:
    return experiment_spec.get("network_type", "RNN-Simple" if experiment_spec["family"] == "RNN-Simple" else "RNN")

def _value_or_default(value: Any, default: Any) -> Any:
    return default if value is None else value

def build_train_configurations(experiment_spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    data_type = experiment_spec.get("data_type", "Spectrogram")
    spectrogram_processing = experiment_spec.get("spectrogram_processing", "magnitude_phase")
    fft_len = int(experiment_spec.get("fft_len", 256))
    output_length = int(experiment_spec.get("output_length", 128))
    batch_size = int(experiment_spec.get("batch_size", 128))
    validation_size = float(experiment_spec.get("validation_size", 0.1))
    learning_rate = float(experiment_spec.get("learning_rate", 0.0001))
    epochs = int(experiment_spec.get("epochs", 1000))
    patience = int(experiment_spec.get("patience", 200))
    factor = float(experiment_spec.get("factor", 0.1))
    optimizer = experiment_spec.get("optimizer", "RMSprop")
    alpha = float(_value_or_default(experiment_spec.get("alpha"), 0.5))
    beta = float(_value_or_default(experiment_spec.get("beta"), alpha))
    gamma = experiment_spec.get("gamma")
    margin = float(_value_or_default(experiment_spec.get("margin"), alpha))

    configurations = {
        "HashNet": {
            "margin": margin,
            "data_type": data_type,
            "spectrogram_processing": spectrogram_processing,
            "fft_len": fft_len,
            "output_length": output_length,
            "batch_size": batch_size,
            "validation_size": validation_size,
            "LearningRate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "factor": factor,
            "optimizer": optimizer,
        },
        "QuadrupletNet": {
            "alpha": alpha,
            "beta": beta,
            "data_type": data_type,
            "spectrogram_processing": spectrogram_processing,
            "fft_len": fft_len,
            "output_length": output_length,
            "batch_size": batch_size,
            "validation_size": validation_size,
            "LearningRate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "factor": factor,
            "optimizer": optimizer,
        },
        "TripletNet": {
            "alpha": alpha,
            "beta": beta,
            "data_type": data_type,
            "spectrogram_processing": spectrogram_processing,
            "fft_len": fft_len,
            "output_length": output_length,
            "batch_size": batch_size,
            "validation_size": validation_size,
            "LearningRate": learning_rate,
            "epochs": epochs,
            "patience": patience,
            "factor": factor,
            "optimizer": optimizer,
        },
    }
    if gamma is not None:
        configurations["QuadrupletNet"]["gamma"] = float(gamma)
    return configurations

def build_model_filename_start(configuration_name: str, experiment_spec: Dict[str, Any]) -> str:
    model_type = resolve_model_type(experiment_spec)
    network_type = resolve_network_type(experiment_spec)
    train_configurations = build_train_configurations(experiment_spec)
    quantization_layer = bool(experiment_spec.get("quantization_layer", False))
    model_count = experiment_spec.get("model_count", 13)
    config = train_configurations[model_type]
    spec_suffix = ""
    if config["data_type"] == "Spectrogram":
        spec_suffix = "_spec" + str(config.get("spectrogram_processing", "magnitude"))
    if model_type == "HashNet":
        return (
            f"{model_type}_{config['data_type']}_FeatureExtractor_{network_type}"
            f"{'_QuantizationLayer' if quantization_layer else ''}"
            f"{spec_suffix}_in{config['fft_len']}_out{config['output_length']}"
            f"_margin{config['margin']}_{config['optimizer']}_lr{config['LearningRate']}"
            f"_{configuration_name}_{model_count}"
        )
    filename = (
        f"{model_type}_{config['data_type']}_FeatureExtractor_{network_type}"
        f"{'_QuantizationLayer' if quantization_layer else ''}"
        f"{spec_suffix}_in{config['fft_len']}_out{config['output_length']}"
        f"_alpha{config['alpha']}_beta{config['beta']}"
    )
    if "gamma" in config:
        filename += f"_gamma{config['gamma']}"
    filename += f"_{config['optimizer']}_lr{config['LearningRate']}_{configuration_name}_{model_count}"
    return filename

def find_existing_model_artifact(models_dir: str, filename_start: str) -> str | None:
    if not os.path.isdir(models_dir):
        return None
    matches = [
        file for file in os.listdir(models_dir)
        if file.startswith(filename_start) and file.endswith(".keras")
    ]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda name: os.path.getmtime(os.path.join(models_dir, name)),
    )[-1]

def prepare_training_dataset(
    signal_type: str = "Sinusoid",
    sample_source: str = "iq",
    augment_with_sionna: bool = True,
    home_dir: str = DEFAULT_HOME_DIR,
    seed: int = 42,
    shuffle_dataset: bool = True,
    generate_plots: bool = False,
    configuration: Dict[str, Any] | None = None,
    sionna_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    set_reproducible(seed)
    node_configurations = build_default_training_node_configurations(signal_type)
    configuration = copy.deepcopy(configuration or node_configurations["OTA-Lab"])
    cache_path = os.path.join(home_dir, "Data", f"{signal_type}_dataset.h5")

    if os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading from {cache_path}")
        source_filter = [SOURCE_OTA_LAB]
        if augment_with_sionna:
            source_filter.append(SOURCE_SIONNA)
        cached = load_cached_dataset(cache_path, source_filter=source_filter)
        data = cached["data"]
        labels = cached["labels"]
        rx = cached["rx"]
        tx = cached["tx"]
        if shuffle_dataset:
            print("Shuffling dataset")
            _shuffler = DatasetHandler.__new__(DatasetHandler)
            data, labels, rx, tx = _shuffler.shuffle_in_groups_of_four(
                data, labels, rx, tx, seed=seed
            )
        if augment_with_sionna and not configuration["config_name"].endswith("+Sionna"):
            configuration["config_name"] += "+Sionna"
        print("Done loading dataset")
    else:
        print(f"[CACHE MISS] {cache_path} not found, loading from remote sources")
        dataset_name = configuration["dataset_name"]
        repo_name = configuration["repo_name"]
        node_id_groups = configuration["node_Ids"]

        for idx, node_ids in enumerate(node_id_groups):
            config_name = configuration["config_name"] + "-" + "".join(str(node) for node in node_ids)
            print("Config name: ", config_name)
            if idx == 0:
                dataset = DatasetHandler(dataset_name, config_name, repo_name)
            else:
                dataset.add_dataset(dataset_name, config_name, repo_name)

        configure_dataset_iq_source(dataset, signal_type, sample_source=sample_source)
        dataset.get_dataframe_Info()
        data, labels, rx, tx = dataset.load_data(shuffle=shuffle_dataset, seed=seed)

        if signal_type == "deltaPulse":
            min_num_samples = int(8192 * 2)
        elif signal_type == "WiFi":
            min_num_samples = int(64) if str(sample_source).strip().lower() == "csi" else int(128)
        else:
            min_num_samples = int(8192)
        data = np.array([example[-min_num_samples:] for example in data])
        hf_num_samples = data.shape[0]

        if augment_with_sionna:
            print("=" * 60)
            print("AUGMENTING with Sionna ray-tracing simulation data")
            print("=" * 60)
            sc = copy.deepcopy(sionna_config or get_default_sionna_config(signal_type))
            sionna_data, sionna_labels, sionna_rx, sionna_tx = generate_multi_scenario_data(
                scenarios=sc["scenarios"],
                num_probes=sc["num_probes"],
                signal_type=sc.get("signal_type", signal_type),
                num_samples=sc["num_samples"],
                backend=sc["backend"],
                noise_voltage=sc["noise_voltage"],
                seed=sc["seed"],
                num_bins=sc.get("num_bins", 2048),
            )
            sionna_target_len = min_num_samples
            if sc["num_samples"] >= sionna_target_len:
                sionna_data = sionna_data[:, -sionna_target_len:]
            else:
                pad_width = sionna_target_len - sc["num_samples"]
                sionna_data = np.pad(
                    sionna_data,
                    ((0, 0), (pad_width, 0)),
                    mode="constant",
                    constant_values=0,
                )
            data = np.concatenate([data, sionna_data], axis=0)
            labels = np.concatenate([labels, sionna_labels], axis=0)
            rx = np.concatenate([rx, sionna_rx], axis=0)
            tx = np.concatenate([tx, sionna_tx], axis=0)

            _shuffler = DatasetHandler.__new__(DatasetHandler)
            data, labels, rx, tx = _shuffler.shuffle_in_groups_of_four(
                data, labels, rx, tx, seed=seed
            )
            sionna_num_samples = sionna_data.shape[0]
            print(
                f"  HuggingFace samples: {hf_num_samples}, "
                f"Sionna samples: {sionna_num_samples}, "
                f"Total: {data.shape[0]}"
            )
            if not configuration["config_name"].endswith("+Sionna"):
                configuration["config_name"] += "+Sionna"

    if generate_plots:
        data_per_scenario = 400
        total_data = data.shape[0]
        number_of_scenarios = total_data // data_per_scenario
        train_plot_dir = os.path.join(home_dir, "Results", "TrainDatasetPlots")
        os.makedirs(train_plot_dir, exist_ok=True)
        for i in range(number_of_scenarios):
            scenario_start_index = int(i * data_per_scenario)
            plot_data_scenario(
                data,
                scenario_start_index,
                os.path.join(train_plot_dir, f"TRAIN_IQ_Spectrogram_scenario_{i}.png"),
                show_plot=False,
            )
            plot_avg_min_max_power_scenario(
                data,
                scenario_start_index,
                data_per_scenario,
                os.path.join(train_plot_dir, f"TRAIN_Avg_Min_Max_Power_scenario_{i}"),
                show_plot=False,
            )

    return {
        "data": data,
        "labels": labels,
        "rx": rx,
        "tx": tx,
        "configuration": configuration,
        "configuration_name": configuration["config_name"],
        "cache_path": cache_path,
    }

def run_training_experiment(
    experiment_spec: Dict[str, Any],
    home_dir: str = DEFAULT_HOME_DIR,
    generate_dataset_plots: bool = False,
) -> Dict[str, Any]:
    spec = copy.deepcopy(experiment_spec)
    spec.setdefault("python_bin", DEFAULT_VENV_PYTHON)
    spec.setdefault("seed", 42)
    model_type = resolve_model_type(spec)
    network_type = resolve_network_type(spec)
    dataset_bundle = prepare_training_dataset(
        signal_type=spec.get("signal_type", "Sinusoid"),
        sample_source=spec.get("sample_source", "iq"),
        augment_with_sionna=bool(spec.get("augment_with_sionna", True)),
        home_dir=home_dir,
        seed=int(spec["seed"]),
        shuffle_dataset=bool(spec.get("shuffle_dataset", True)),
        generate_plots=generate_dataset_plots,
        configuration=spec.get("training_configuration"),
        sionna_config=spec.get("sionna_config"),
    )
    filename_start = build_model_filename_start(dataset_bundle["configuration_name"], spec)
    models_dir = os.path.join(home_dir, "Models")
    existing_model = find_existing_model_artifact(models_dir, filename_start)
    if existing_model:
        return {
            "family": spec["family"],
            "model_type": model_type,
            "network_type": network_type,
            "model_name": existing_model,
            "model_path": os.path.join(models_dir, existing_model),
            "configuration_name": dataset_bundle["configuration_name"],
            "reused_existing": True,
            "train_configurations": build_train_configurations(spec)[model_type],
            "quantization_layer": bool(spec.get("quantization_layer", False)),
            "python_bin": spec["python_bin"],
        }

    feature_extractor = train_channel_feature_extractor(
        dataset_bundle["data"],
        dataset_bundle["labels"],
        build_train_configurations(spec),
        model_type,
        network_type=network_type,
        model_name=filename_start,
        seed=int(spec["seed"]),
        quantization_layer=bool(spec.get("quantization_layer", False)),
        home_dir=home_dir,
    )
    saved_model = find_existing_model_artifact(models_dir, filename_start)
    if saved_model is None:
        raise FileNotFoundError(f"Training completed but no saved .keras artifact matched prefix: {filename_start}")
    return {
        "family": spec["family"],
        "model_type": model_type,
        "network_type": network_type,
        "model_name": saved_model,
        "model_path": os.path.join(models_dir, saved_model),
        "configuration_name": dataset_bundle["configuration_name"],
        "reused_existing": False,
        "train_configurations": build_train_configurations(spec)[model_type],
        "quantization_layer": bool(spec.get("quantization_layer", False)),
        "python_bin": spec["python_bin"],
        "feature_extractor_name": feature_extractor.name if feature_extractor is not None else None,
    }

def configure_dataset_iq_source(dataset, signal_type, sample_source="auto"):
    """
    Ensure DatasetHandler has `I`/`Q` columns for complex conversion.

    sample_source options:
      - "auto": WiFi -> chan_est_samples, others -> existing I/Q (or iq_I/Q fallback)
      - "chan_est_samples": use chan_est_samples_I/Q
      - "csi": use csi_I/Q
      - "iq": use existing I/Q (or iq_I/Q fallback)
    """
    if dataset is None or dataset.dataFrame is None:
        raise ValueError("Dataset is not loaded.")

    df = dataset.dataFrame
    source = str(sample_source).strip().lower()
    if source == "auto":
        source = "chan_est_samples" if signal_type == "WiFi" else "iq"

    source_to_columns = {
        "chan_est_samples": ("chan_est_samples_I", "chan_est_samples_Q"),
        "chan_est": ("chan_est_samples_I", "chan_est_samples_Q"),
        "csi": ("csi_I", "csi_Q"),
        "iq": ("I", "Q"),
    }
    if source not in source_to_columns:
        raise ValueError(
            f"Unsupported sample_source='{sample_source}'. "
            "Use one of: auto, chan_est_samples, csi, iq."
        )

    i_col, q_col = source_to_columns[source]
    if source == "iq" and not {"I", "Q"}.issubset(set(df.columns)) and {"iq_I", "iq_Q"}.issubset(set(df.columns)):
        i_col, q_col = "iq_I", "iq_Q"

    required_cols = {i_col, q_col}
    missing_cols = sorted(list(required_cols - set(df.columns)))
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns for sample_source='{source}': {missing_cols}"
        )

    df = df.copy()
    df["I"] = df[i_col]
    df["Q"] = df[q_col]
    dataset.dataFrame = df
    print(f"Using {i_col}/{q_col} as IQ input for training.")

def train_channel_feature_extractor(data, labels, train_configurations, model_type, network_type="ResNet", model_name="", seed: int = 42, quantization_layer=False, home_dir: str = DEFAULT_HOME_DIR):
    '''
    train_feature_extractor trains an RFF extractor using triplet loss.
    
    INPUT: 
        FILE_PATH is the path of training dataset.
        
        DEV_RANGE is the label range of LoRa devices to train the RFF extractor.
        
        PKT_RANGE is the range of packets from each device to train the RFF extractor.
        
        SNR_RANGE is the SNR range used in data augmentation. 
        
    RETURN:
        FEATURE_EXTRACTOR is the RFF extractor which can extract features from
        channel-independent spectrograms.
    '''
        
    # Ensure reproducibility for any TF/NumPy/Python randomness within training
    set_reproducible(seed)
    
    if train_configurations[model_type]['data_type'] == "IQ":
        ChannelIQObj = ChannelIQ()
        use_rnn_format = network_type in ("RNN", "RNN-Simple")
        data = ChannelIQObj.channel_iq(data, rnn_format=use_rnn_format)
    elif train_configurations[model_type]['data_type'] == "Polar":
        ChannelPolarObj = ChannelPolar()
        data = ChannelPolarObj.channel_polar(data)
    elif train_configurations[model_type]['data_type'] == "Spectrogram":
        ChannelSpectrogramObj = ChannelSpectrogram()
        spectrogram_processing = train_configurations[model_type].get("spectrogram_processing", "magnitude")
        normalize = True
        data = ChannelSpectrogramObj.channel_spectrogram(
            data,
            train_configurations[model_type]['fft_len'],
            processing=spectrogram_processing,
            normalize=normalize
        )

    #NetObj =  TripletNet_Channel()
    # NetObj = QuadrupletNet_Channel()
    
    # NetObj = ResNet_QuadrupletNet_Channel()
    output_length = train_configurations[model_type]['output_length']
    # Build model graph/variables on requested device context.
    with training_device_context():
        # Create an RFF extractor.
        if model_type == "HashNet":
            # HashNet uses siamese architecture
            if network_type == "ResNet":
                NetObj = ResNet_HashNet_Channel()
            elif network_type == "RNN":
                NetObj = RNN_HashNet_Channel()
            else:
                raise ValueError(f"HashNet supports only ResNet or RNN network_type, got {network_type}")
            
            feature_extractor = NetObj.feature_extractor(data.shape, output_length)
            
            # Create the siamese net using the RFF extractor.
            margin = train_configurations[model_type].get('margin', 0.3)
            net = NetObj.siammese_net(feature_extractor, m=margin)
        elif model_type == "TripletNet":
            # TripletNet architecture
            if network_type == "RNN":
                NetObj = RNN_TripletNet_Channel()
            elif network_type == "ResNet":
                NetObj = TripletNet_Channel()
            else:
                raise ValueError(f"TripletNet supports only ResNet or RNN network_type, got {network_type}")

            feature_extractor = NetObj.feature_extractor(data.shape, output_length)
            net = NetObj.create_triplet_net(feature_extractor, train_configurations[model_type]['alpha'])
        else:
            # QuadrupletNet architecture
            if network_type == "ResNet":
                NetObj = ResNet_QuadrupletNet_Channel()
            elif network_type == "FeedForward":
                NetObj = FeedForward_QuadrupletNet_Channel()
            elif network_type == "RNN":
                NetObj = RNN_QuadrupletNet_Channel()
            elif network_type == "RNN-Simple":
                NetObj = RNN_QuadrupletNet_Channel_Simple()
            elif network_type == "Transformer":
                NetObj = Transformer_QuadrupletNet_Channel()
            elif network_type == "AE":
                NetObj = AE_QuadrupletNet_Channel()
            else:
                raise ValueError(f"Unknown network_type: {network_type}")
            
            if quantization_layer:
                feature_extractor = NetObj.build_quantized_extractor(data.shape, output_length)
            else:
                feature_extractor = NetObj.feature_extractor(data.shape, output_length)
            
            # Create the quadruplet net using the RFF extractor.
            loss_type = "KDR" if quantization_layer else ""
            net = NetObj.create_quadruplet_net(
                feature_extractor, 
                train_configurations[model_type]['alpha'], 
                train_configurations[model_type]['beta'], 
                train_configurations[model_type]['gamma'] if 'gamma' in train_configurations[model_type] else None,
                loss_type=loss_type
            )

    # Create callbacks during training. The training stops when validation loss 
    # does not decrease for 30 epochs.
    patience = train_configurations[model_type]['patience']
    
    early_stop = EarlyStopping(
                                monitor='val_loss',
                                min_delta=1e-4,
                                patience=patience,
                                mode='min',
                                restore_best_weights=True,
                                verbose=1
                                )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    min_delta = 0.0001, 
                                    factor = train_configurations[model_type]['factor'], 
                                    patience = int(math.ceil(patience/2)), 
                                    verbose=1
                                    )
    
    ResultsDir = os.path.join(home_dir, "Results") + "/"
    ModelsDir = os.path.join(home_dir, "Models") + "/"
    
    # Model checkpoint
    checkpoint_filename = os.path.join(ResultsDir, f"model_checkpoint_{model_name}.weights.h5")
    checkpoint = ModelCheckpoint(checkpoint_filename, 
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min',
                                 save_weights_only=True)
    
    callbacks = [early_stop, reduce_lr, checkpoint]
    
    validation_size= train_configurations[model_type]['validation_size']
    
    # Split dataset on quadruplet boundaries so [AB, AE, BA, BE] ordering is preserved.
    total_len = data.shape[0]
    valid_len = int(total_len * validation_size)
    valid_len = max(4, (valid_len // 4) * 4)
    train_len = total_len - valid_len
    train_len = (train_len // 4) * 4

    if train_len < 4 or valid_len < 4:
        raise ValueError(
            f"Not enough data for quadruplet-aligned split. total={total_len}, "
            f"train_len={train_len}, valid_len={valid_len}"
        )

    used_len = train_len + valid_len
    if used_len < total_len:
        dropped = total_len - used_len
        print(f"Dropping {dropped} samples to preserve quadruplet alignment.")

    data_train = data[:train_len]
    label_train = labels[:train_len]
    data_valid = data[train_len:train_len + valid_len]
    label_valid = labels[train_len:train_len + valid_len]
    del data, labels
    
    batch_size = train_configurations[model_type]['batch_size']
    # Create the trainining generator.
    train_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_train, 
                                                        label_train,
                                                        seed=seed)
    
    # Create the validation generator.
    valid_generator = NetObj.create_generator_channel(batch_size, 
                                                        data_valid, 
                                                        label_valid,
                                                        seed=seed + 1 if isinstance(seed, int) else None)
    
    # Use the RMSprop optimizer for training.
    LearningRate = train_configurations[model_type]['LearningRate']
    optimizer = train_configurations[model_type]['optimizer']
    with training_device_context():
        if optimizer == "Adam":
            opt = Adam(learning_rate=LearningRate)
        elif optimizer == "RMSprop":
            opt = RMSprop(learning_rate=LearningRate)
        elif optimizer == "SGD":
            opt = SGD(learning_rate=LearningRate)

        net.compile(
            loss = identity_loss,
            #metrics=['accuracy'],
            optimizer = opt)

    print("Training data:", data_train.shape)
    print("Validation data:", data_valid.shape)
    # Start training.
    history = net.fit(train_generator,
                        steps_per_epoch = data_train.shape[0]//batch_size,
                        epochs = train_configurations[model_type]['epochs'],
                        validation_data = valid_generator,
                        validation_steps = data_valid.shape[0]//batch_size,
                        verbose=1, 
                        callbacks = callbacks)
    # Prefer checkpoint-selected weights when available.
    if os.path.exists(checkpoint_filename):
        net.load_weights(checkpoint_filename)
    # Save the history and reproducibility metadata
    with h5py.File(ResultsDir+"History_"+model_name, "w") as f:
                        f.create_dataset("train_loss",data=history.history['loss'])
                        f.create_dataset("validation_loss",data=history.history['val_loss'])
                        f.close()
                        
    timestamp = int(time.time())
    keras_filename = model_name +'_'+str(timestamp)+'.keras'
    
    # feature_extractor.save(ModelsDir+filename)
    # print("Saving file: ", filename)
    try:
        save_feature_extractor_keras(feature_extractor, ModelsDir + keras_filename)
        print("Saving Keras file: ", keras_filename)
    except Exception as exc:
        print("WARNING: .keras save failed:", exc)
    
    return feature_extractor

if __name__ == "__main__":
    homeDir = "/home/Research/POWDER/"
    ModelsDir = homeDir+"Models/"
    # Central seed for full reproducibility of data shuffling, initialization, and training
    
    signal_type = "Sinusoid" # Sinusoid, PN-Sequence, deltaPulse, WiFi
    sample_source = "iq"  # auto, chan_est_samples, csi, iq
    augment_with_sionna = True  # Set True to append Sionna ray-tracing data

    # ── Sionna ray-tracing augmentation config (used when augment_with_sionna=True)
    sionna_config = {
        "signal_type": signal_type,       # "Sinusoid" or "deltaPulse"
        "num_probes": 100,                # probes per scenario
        "num_samples": 8192,              # IQ samples per probe
        "num_bins": 1024,                 # IFFT size for deltaPulse probe
        "backend": "gnuradio",              # "numpy" (fast, no GNURadio) or "gnuradio"
        "noise_voltage": 0.01,
        "seed": 42,
        "scenarios": [
            # Each entry is [Alice, Bob, Eve] using POWDER node names or IDs
            ["EBC", "Guest House", "Moran"],
            ["EBC", "Guest House", "USTAR"],
            ["EBC", "Moran", "Guest House"],
            ["EBC", "Moran", "USTAR"],
            ["EBC", "USTAR", "Guest House"],
            ["EBC", "USTAR", "Moran"],
            ["Guest House", "Moran", "EBC"],
            ["Guest House", "Moran", "USTAR"],
            ["Guest House", "USTAR", "Moran"],
            ["Guest House", "USTAR", "EBC"],
            ["USTAR", "Moran", "EBC"],
            ["USTAR", "Moran", "Guest House"],
        ],
    }

    node_Ids = {"Sinusoid":
                    {"OTA-Lab": [[1,2,3],
                            [1,4,5],
                            [1,4,8],
                            [2,4,3],
                            [4,2,5],
                            [4,2,8],
                            [4,8,5],
                            [5,7,8],
                            [5,8,7],
                            [8,4,1],
                            [8,5,1],
                            # [8,5,4]
                            ]
                    ,
                    "OTA-Dense": [#[1,2,3],
                                #[1,2,5],
                               # [1,3,2],
                               # [4,3,5]
                    ]
                    },
                "PN-Sequence": {
                    "OTA-Lab": [],
                    "OTA-Dense": []
                },
                "deltaPulse": {
                    "OTA-Lab": [
                                    # [5,6,1],[5,6,2],[5,6,3],  # Testing scenarios
                                    # [5,6,4],[5,6,7],[5,6,8],  # Testing scenarios
                                    # [5,7,1],[5,7,3],[5,7,2],# May need to remove due to low signal power
                                    # [5,7,4],[5,7,6],[5,7,8],# May need to remove due to low signal power
                                    [5,8,2],[5,8,3],
                                    # [5,8,4],
                                    [5,8,6],[5,8,7],[6,7,2],
                                    [6,7,3],[6,7,4],[6,7,5],
                                    [6,7,8],
                                    # [6,8,2],[6,8,3],# May need to remove due to low signal power
                                    # [6,8,4],[6,8,5],[6,8,7],# May need to remove due to low signal power
                                    # [7,8,2],[7,8,3],[7,8,4],# May need to remove due to low signal power
                                    # [7,8,5],[7,8,6]# May need to remove due to low signal power
                                    #     #Scenarios avoided due to low signal power
                                ],
                    "OTA-Dense": [
                                    # [1,2,3],[1,2,4],[1,3,2],
                                    # [1,3,4],[1,4,2],[1,4,3],
                                    # [2,3,1],[2,3,4],[2,4,1],
                                    # [2,4,3],[3,4,1],[3,4,2]
                                ]
                },
                "WiFi": {
                    "OTA-Lab": [
                        [5,6,2],
                        [5,6,3],
                        [5,6,4],
                        [5,6,7],
                        [5,6,8],
                        [5,7,2],
                        [5,7,3],
                        [5,7,4],
                        [5,7,6],
                        [5,7,8],
                        [5,8,2],
                        [5,8,3],
                        # [5,8,4],
                        [5,8,6],
                        [5,8,7],
                        [6,7,2],
                    ],
                    "OTA-Dense": []
                },
    }
    node_configurations = {
        'OTA-Lab': {
            'dataset_name': 'Key-Generation',
            'config_name': signal_type+'-Powder-OTA-Lab-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': node_Ids[signal_type]["OTA-Lab"]               
        },
        'OTA-Dense': {
            'dataset_name': 'Key-Generation',
            'config_name': signal_type+'-Powder-OTA-Dense-Nodes',
            'repo_name': 'CAAI-FAU',
            'node_Ids': node_Ids[signal_type]["OTA-Dense"]
        }
    }
    
    REPRO_SEED = 42
    set_reproducible(REPRO_SEED)
    shuffle_dataset = True

    # ── 1. Try loading from local HDF5 cache ────────────────────────────
    cache_path = os.path.join(homeDir, "Data", f"{signal_type}_dataset.h5")
    configuration = node_configurations['OTA-Lab']

    if os.path.exists(cache_path):
        print(f"[CACHE HIT] Loading from {cache_path}")
        source_filter = [SOURCE_OTA_LAB]
        if augment_with_sionna:
            source_filter.append(SOURCE_SIONNA)
        cached = load_cached_dataset(cache_path, source_filter=source_filter)
        data = cached["data"]
        labels = cached["labels"]
        rx = cached["rx"]
        tx = cached["tx"]

        if shuffle_dataset:
            print("Shuffling dataset")
            _shuffler = DatasetHandler.__new__(DatasetHandler)
            data, labels, rx, tx = _shuffler.shuffle_in_groups_of_four(
                data, labels, rx, tx, seed=REPRO_SEED)

        if augment_with_sionna:
            configuration["config_name"] += "+Sionna"
        print("Done loading dataset")
    else:
        print(f"[CACHE MISS] {cache_path} not found, loading from remote sources")
        # ── 1b. Load from HuggingFace ─────────────────────────────────
        dataset_name = configuration['dataset_name']
        repo_name = configuration['repo_name']
        node_Ids = configuration['node_Ids']

        for idx, node_ids in enumerate(node_Ids):
            config_name = configuration['config_name']+"-"+"".join(str(node) for node in node_ids)
            print("Config name: ", config_name)
            if idx == 0:
                dataset = DatasetHandler(dataset_name, config_name, repo_name)
            else:
                dataset.add_dataset(dataset_name, config_name, repo_name)

        configure_dataset_iq_source(dataset, signal_type, sample_source=sample_source)
        dataset.get_dataframe_Info()
        data, labels, rx, tx = dataset.load_data(shuffle=shuffle_dataset, seed=REPRO_SEED)

        print("First example number of samples: ", data[0].shape[0])
        print("Last example number of samples: ", data[-1].shape[0])
        avg_num_samples = np.mean([example.shape[0] for example in data])
        print("Average number of samples per example: ", avg_num_samples)
        min_num_samples = np.min([example.shape[0] for example in data])
        print("Example with the least number of samples: ", min_num_samples)
        max_num_samples = np.max([example.shape[0] for example in data])
        print("Example with the most number of samples: ", max_num_samples)
        if signal_type == "deltaPulse":
            min_num_samples = int(8192 * 2)
        elif signal_type == "WiFi":
            if str(sample_source).strip().lower() == "csi":
                min_num_samples = int(64)
            else:
                min_num_samples = int(128)
        else:
            min_num_samples = int(8192)
        data = np.array([example[-min_num_samples:] for example in data])
        hf_num_samples = data.shape[0]

        # ── 2. Optionally augment with Sionna ray-tracing data ──────────
        if augment_with_sionna:
            print("=" * 60)
            print("AUGMENTING with Sionna ray-tracing simulation data")
            print("=" * 60)
            sc = sionna_config
            sionna_data, sionna_labels, sionna_rx, sionna_tx = generate_multi_scenario_data(
                scenarios=sc["scenarios"],
                num_probes=sc["num_probes"],
                signal_type=sc.get("signal_type", signal_type),
                num_samples=sc["num_samples"],
                backend=sc["backend"],
                noise_voltage=sc["noise_voltage"],
                seed=sc["seed"],
                num_bins=sc.get("num_bins", 2048),
            )
            sionna_target_len = min_num_samples
            if sc["num_samples"] >= sionna_target_len:
                sionna_data = sionna_data[:, -sionna_target_len:]
            else:
                pad_width = sionna_target_len - sc["num_samples"]
                sionna_data = np.pad(
                    sionna_data, ((0, 0), (pad_width, 0)),
                    mode="constant", constant_values=0,
                )
            data = np.concatenate([data, sionna_data], axis=0)
            labels = np.concatenate([labels, sionna_labels], axis=0)
            rx = np.concatenate([rx, sionna_rx], axis=0)
            tx = np.concatenate([tx, sionna_tx], axis=0)

            _shuffler = DatasetHandler.__new__(DatasetHandler)
            data, labels, rx, tx = _shuffler.shuffle_in_groups_of_four(
                data, labels, rx, tx, seed=REPRO_SEED)
            sionna_num_samples = sionna_data.shape[0]
            print(f"  HuggingFace samples: {hf_num_samples}, "
                  f"Sionna samples: {sionna_num_samples}, "
                  f"Total: {data.shape[0]}")
            configuration["config_name"] += "+Sionna"

    print("Data shape: ", data.shape)
    print("Labels shape: ", labels.shape)
    print("RX shape: ", rx.shape)
    print("TX shape: ", tx.shape)

    DataPerScenario = 400
    TotalData = data.shape[0]
    numberOfScenarios = TotalData//DataPerScenario
    ScenarioToTest = 0
    print("Scenario to test: ", ScenarioToTest)
    print("Number of scenarios: ", numberOfScenarios)
    print("Data per scenario: ", DataPerScenario)
    print("Total data: ", TotalData)
    print("Data per scenario: ", DataPerScenario)
    
    train_plot_dir = os.path.join(homeDir, "Results", "TrainDatasetPlots")
    os.makedirs(train_plot_dir, exist_ok=True)
    for i in range(numberOfScenarios):
        ScenarioStartIndex = int(i*DataPerScenario)
        plot_data_scenario(
            data,
            ScenarioStartIndex,
            os.path.join(train_plot_dir, "TRAIN_IQ_Spectrogram_scenario_" + str(i) + ".png"),
            show_plot=False,
        )
        # plot_avg_power_scenario(data, ScenarioStartIndex, DataPerScenario, "XTEST_Avg_Power_scenario_"+str(i)+".png")
        plot_avg_min_max_power_scenario(
            data,
            ScenarioStartIndex,
            DataPerScenario,
            os.path.join(train_plot_dir, "TRAIN_Avg_Min_Max_Power_scenario_" + str(i)),
            show_plot=False,
        )
    # exit()
    # Locate locations for delta pulses
    # def locate_delta_pulse(example_data):
    #     """
    #     Locate the positions of delta pulses in IQ data.
        
    #     Parameters:
    #     -----------
    #     example_data : np.ndarray
    #         Complex-valued IQ signal data
            
    #     Returns:
    #     --------
    #     np.ndarray
    #         Array of indices where delta pulses are located
    #     """
    #     # Calculate magnitude for complex IQ data
        
    #     magnitude = np.abs(example_data)
        
    #     # Use percentile-based threshold to detect spikes (delta pulses)
    #     # Delta pulses are typically much higher than the median/background
    #     # Using 95th percentile as threshold to catch significant spikes
    #     threshold = np.percentile(magnitude, 99.95)
        
    #     # Alternative: Use median + multiple of MAD (Median Absolute Deviation) for robustness
    #     median = np.median(magnitude)
    #     mad = np.median(np.abs(magnitude - median))
    #     print("Median: ", median)
    #     print("MAD: ", mad)
    #     print("Magnitude: ", magnitude)
    #     # Get the max value of the magnitude
    #     max_value = np.max(magnitude)
    #     print("Max value: ", max_value)
        
    #     # get the 20 max values of the magnitude
    #     max_values = np.sort(magnitude)[-20:]
    #     print("Max values: ", max_values)
        
    #     threshold = magnitude #median + 1 * mad
    #     print("Threshold: ", threshold)
        
    #     # Find indices where magnitude exceeds threshold
    #     delta_pulse_indices = np.where(magnitude > threshold)[0]
        
    #     return delta_pulse_indices
    # # Print number of delta pulses
    
    # def locate_delta_pulse_energy(iq_samples, threshold_factor=7, window_size=256):
    #     # 1. Energy
    #     energy = np.abs(iq_samples)**2

    #     # 2. Smooth energy
    #     window = window_size
    #     energy_smooth = np.convolve(
    #         energy,
    #         np.ones(window)/window,
    #         mode='same'
    #     )

    #     # 3. Threshold
    #     noise_floor = np.median(energy_smooth)
    #     threshold = noise_floor * threshold_factor

    #     # 4. Detection
    #     is_signal = energy_smooth > threshold

    #     # 5. Rising edges = burst starts
    #     edges = np.diff(is_signal.astype(int))
    #     start_indices = np.where(edges == 1)[0]
    #     # get the end indices of the pulses
    #     end_indices = np.where(edges == -1)[0]
    #     # get the duration of the pulses
    #     # pulse_durations = start_indices - end_indices
    #     # print("Pulse durations: ", pulse_durations)
    #     # # get the average pulse duration
    #     # average_pulse_duration = np.mean(pulse_durations)
    #     # print("Average pulse duration: ", average_pulse_duration)
    #     # # get the median pulse duration
    #     # median_pulse_duration = np.median(pulse_durations)
    #     # print("Median pulse duration: ", median_pulse_duration)

    #     print("Detected burst start indices:")
    #     print(start_indices)
    #     return start_indices
    # window_size = 128
    # threshold_factor = 5
    # # print("Number of delta pulses: ", len(locate_delta_pulse(example_data)))
    # pulse_indices = locate_delta_pulse_energy(example_data, threshold_factor=threshold_factor, window_size=window_size)
    # print("Number of delta pulses: ", len(pulse_indices))
    
    
    
    # plt.imshow(arr, aspect='auto', cmap='jet')
    # plt.title("Spectrogram")
    # plt.xlabel("Frequency")
    # plt.ylabel("Time")
    # plt.colorbar()
    # plt.show()
    # plt.savefig("XTEST_IQ_Spectrogram_class.png")
    
    # Plot each of the  pulses in the time domain
    # Each pulse will have N samples and the pulse will be at the center of the N samples
    # Create a figure with D delta pulses
    # D = len(pulse_indices)
    # fig, axes = plt.subplots(D, 1, figsize=(10, 10))
    # pulses = []
    # # window_size = 256
    # for i, pulse_index in enumerate(pulse_indices):
    #     x_start = pulse_index
    #     x_end = pulse_index+window_size
    #     pulse = example_data[int(x_start):int(x_end)]
    #     pulses.append(pulse)
    #     axes[i].plot(pulse.real, label='Real')
    #     axes[i].plot(pulse.imag, label='Imaginary')
    #     # Mark the pulse index with a vertical line
    #     # Change the x-axis to the pulse index
    #     # axes[i].set_xlim(x_start, x_end)
    #     # axes[i].axvline(x=pulse_index, color='red', linestyle='--')
    #     axes[i].set_title("Pulse")
    #     axes[i].set_xlabel("Sample Number")
    #     axes[i].set_ylabel("Amplitude")
    #     # Set x axis ticks to start and end of the pulse
    #     # axes[i].set_xticks([int(x_start), int(x_end)])
    #     xticks = np.linspace(int(x_start), int(x_end))
    #     # axes[i].set_xticks(xticks)
    #     axes[i].set_xticklabels([int(x) for x in xticks])
    #     # mark the pulse index with a vertical line
    #     # indexMarker = window_size/2
    #     # print("Index marker: ", indexMarker)
    #     # axes[i].axvline(x=indexMarker, color='red', linestyle='--')
    #     # axes[i].legend()
    # # Set independent x axis for each subplot
    # plt.subplots_adjust(hspace=1)
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("XTEST_IQ_Pulses.png")
    
    # Create a spectrogram of the pulses
    # Apply FFT to each pulse in pulses
    # Create a new figure with single plot
    
    # spectrograms = []
    # for pulse in pulses:
    #     spectrogram = np.abs(np.fft.fftshift(np.fft.fft(pulse, axis=0), axes=0))
    #     spectrograms.append(spectrogram)
    # # Plot the spectrograms
    # fig, axes = plt.subplots(1, 1, figsize=(10, 10))
    # plt.imshow(spectrograms, aspect='auto', cmap='jet')
    # plt.title("Spectrograms")
    # plt.xlabel("Frequency")
    # plt.ylabel("Time")
    # plt.colorbar()
    # plt.savefig("XTEST_IQ_Pulses_Spectrograms_per_pulse.png")
    # exit()
    model_type = "QuadrupletNet" # "HashNet", "QuadrupletNet", "TripletNet"
    data_type = "Spectrogram"
    if str(sample_source).strip().lower() == "csi":
        # For CSI input, use raw complex examples (no FFT/spectrogram).
        data_type = "IQ"
    # data_types = ["IQ", "Polar", "Spectrogram"] # ["IQ", "Polar", "Spectrogram"]
    spectrogram_processing = "magnitude_phase"  # "magnitude", "complex", "magnitude_phase"
    batch_size = 128 
    if signal_type == "Sinusoid":
        fft_len = int(256)
    elif signal_type == "deltaPulse":
        fft_len = int(2048)
    elif signal_type == "WiFi":
        fft_len = int(64)
    else:
        fft_len = int(256)
    patience = 200
    maxEpochs = 1000
    val_size = 0.1
    factor = 0.1
    alphas = [0.5] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    betas = [alphas[0]] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    gammas = [0.2] # [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    FFT_lengths = [fft_len]
    output_lengths = [128] # Need to define the output length and why based on 255 constraint (bytes)
    optimizers = ["RMSprop"] # "RMSprop", "SGD", "Adam"
    network_types = ["RNN"] # ["ResNet", "FeedForward", "RNN", "RNN-Simple", "Transformer", "AE"]
    quantization_layer = True
    for fft_len in FFT_lengths:
        for output_length in output_lengths:
            for network_type in network_types:
                for a in alphas:
                    for b in betas:
                        for g in gammas:
                            for o in optimizers:
                                # b = a
                                if o == "RMSprop":
                                    lr = 0.0001
                                elif o == "SGD":
                                    lr = 0.1
                                elif o == "Adam":
                                    lr = 0.0001
                                print("Alpha: ", a, "Beta: ", b, "Gamma: ", g, "Optimizer: ", o, "Network Type: ", network_type)
                                train_configurations = {
                                    "HashNet": {
                                        "margin": a,
                                        "data_type": data_type,
                                        "spectrogram_processing": spectrogram_processing,
                                        "fft_len": fft_len,
                                        "output_length": output_length,
                                        "batch_size": batch_size,
                                        "validation_size": val_size,
                                        "LearningRate": lr,
                                        "epochs": maxEpochs,
                                        "patience": patience,
                                        "factor": factor,
                                        "optimizer": o,
                                    },
                                    "QuadrupletNet": {
                                        "alpha": a,
                                        "beta": b,
                                        # "gamma": g,
                                        "data_type": data_type,
                                        "spectrogram_processing": spectrogram_processing,
                                        "fft_len": fft_len,
                                        "output_length": output_length,
                                        "batch_size": batch_size,
                                        "validation_size": val_size,
                                        "LearningRate": lr,
                                        "epochs": maxEpochs,
                                        "patience": patience,
                                        "factor": factor,
                                        "optimizer": o,
                                    },
                                    "TripletNet": {
                                        "alpha": a,
                                        "beta": b,
                                        "data_type": data_type,
                                        "spectrogram_processing": spectrogram_processing,
                                        "fft_len": fft_len,
                                        "output_length": output_length,
                                        "batch_size": batch_size,
                                        "validation_size": val_size,
                                        "LearningRate": lr,
                                        "epochs": maxEpochs,
                                        "patience": patience,
                                        "factor": factor,
                                        "optimizer": o
                                    }
                                }
                                model_count = 13
                                if model_type == "HashNet":
                                    spec_suffix = ""
                                    if train_configurations[model_type]["data_type"] == "Spectrogram":
                                        spec_suffix = "_spec" + str(train_configurations[model_type].get("spectrogram_processing", "magnitude"))
                                    filename_start = model_type+'_'+train_configurations[model_type]['data_type'] \
                                            +'_FeatureExtractor_'+network_type \
                                            +('_QuantizationLayer' if quantization_layer else "") \
                                            +spec_suffix \
                                            +'_in'+str(train_configurations[model_type]["fft_len"]) \
                                            +'_out'+str(train_configurations[model_type]["output_length"]) \
                                            +'_margin'+str(train_configurations[model_type]["margin"]) \
                                            +'_'+train_configurations[model_type]['optimizer'] \
                                            +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                                            +'_'+configuration["config_name"]+'_'+str(model_count)
                                else:
                                    spec_suffix = ""
                                    if train_configurations[model_type]["data_type"] == "Spectrogram":
                                        spec_suffix = "_spec" + str(train_configurations[model_type].get("spectrogram_processing", "magnitude"))
                                    filename_start = model_type+'_'+train_configurations[model_type]['data_type'] \
                                            +'_FeatureExtractor_'+network_type \
                                            +('_QuantizationLayer' if quantization_layer else "") \
                                            +spec_suffix \
                                            +'_in'+str(train_configurations[model_type]["fft_len"]) \
                                            +'_out'+str(train_configurations[model_type]["output_length"]) \
                                            +'_alpha'+str(train_configurations[model_type]["alpha"]) \
                                            +'_beta'+str(train_configurations[model_type]["beta"]) \
                                            +('_gamma'+str(train_configurations[model_type]["gamma"]) if "gamma" in train_configurations[model_type] else "") \
                                            +'_'+train_configurations[model_type]['optimizer'] \
                                            +'_lr'+str(train_configurations[model_type]["LearningRate"]) \
                                            +'_'+configuration["config_name"]+'_'+str(model_count)
                                
                                model_exisits = False
                                # Check if there is a file that starts with the same name
                                for file in os.listdir(ModelsDir):
                                    if file.startswith(filename_start):
                                        model_exisits = True
                                        filename = file
                                        break
                                # if model_exisits:
                                #     print("Model already exists")
                                #     print("Skipping: ", filename)
                                #     # Plot the history of the model located in results directory
                                #     history_file = homeDir+"Results/History_"+filename
                                #     history = h5py.File(history_file, "r")
                                #     train_loss = history["train_loss"][:]
                                #     validation_loss = history["validation_loss"][:]
                                #     plt.plot(train_loss)
                                #     plt.plot(validation_loss)
                                #     plt.title(filename+" Model History")
                                #     plt.ylabel("Loss")
                                #     plt.xlabel("Epoch")
                                #     plt.legend(["Train", "Validation"], loc="upper left")
                                #     plt.show()
                                #     plt.savefig(homeDir+"Results/"+filename_start+"_history.png")
                                #     continue
                                # else:
                                #     print("Training model: ", filename_start)
                                    
                                feature_extractor = train_channel_feature_extractor(
                                    data,
                                    labels,
                                    train_configurations,
                                    model_type,
                                    network_type,
                                    filename_start,
                                    seed=REPRO_SEED,
                                    quantization_layer=quantization_layer
                                    
                                )