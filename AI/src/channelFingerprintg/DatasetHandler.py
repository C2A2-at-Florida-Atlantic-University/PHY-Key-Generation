import numpy as np
from datasets import Dataset, DatasetDict, DownloadMode
import datasets
import pandas as pd
import matplotlib.pyplot as plt
from DatasetGenerator import DatasetGenerator
import numpy as np

from scipy import signal

class DatasetHandler():
    def __init__(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.repo_name = repo_name
        self.dataFrame = None
        self.load_dataset()
        
    def load_dataset(self):
        path = self.repo_name+"/"+self.dataset_name
        print("Path: ", path)
        print("Config name: ", self.config_name)
        self.dataFrame = datasets.load_dataset(
            path,
            self.config_name,
            download_mode=DownloadMode.FORCE_REDOWNLOAD
        )
        # self.get_dataframe_Info()
        if isinstance(self.dataFrame, DatasetDict):
            self.dataFrame = pd.concat([self.dataFrame[key].to_pandas() for key in self.dataFrame.keys()])
        # Check if dataframe is a dataset
        elif isinstance(self.dataFrame, Dataset):
            # convert to pandas dataframe
            self.dataFrame = self.dataFrame.to_pandas()
            
    def add_dataset(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        '''Add a new dataset to the existing dataset.'''
        new_dataset = datasets.load_dataset(repo_name+"/"+dataset_name, config_name)
        max_id = self.dataFrame["ids"].max() # Get the maximum ids number in the current dataset
        new_dataset = new_dataset.map(lambda x: {"ids": x["ids"] + max_id}) # Add the maximum ids number to the ids number of the new dataset
        if isinstance(new_dataset, DatasetDict):
            new_dataset = pd.concat([new_dataset[key].to_pandas() for key in new_dataset.keys()])
        elif isinstance(new_dataset, Dataset):
            new_dataset = new_dataset.to_pandas()
        self.dataFrame = pd.concat([self.dataFrame, new_dataset], ignore_index=True)
        
    def get_dataframe_Info(self):
        print("DataFrame Info:")
        print(self.dataFrame.info())
        print("DataFrame Shape:", self.dataFrame.shape)
        print("DataFrame Columns:",self. dataFrame.columns)
        print("DataFrame Head:")
        print(self.dataFrame.head())
        print("DataFrame Description:")
        print(self.dataFrame.describe())
        
    def separate_iq_samples(self, data):
        '''Separate the IQ samples into I and Q components.'''
        # I is the second half of the samples
        I = data[len(data)//2:]
        # Q is the first half of the samples
        Q = data[:len(data)//2]
        return I, Q
    
    def convert_to_complex(self):
        '''Convert the loaded data to complex IQ samples.'''
        I = self.dataFrame["I"].to_numpy()
        Q = self.dataFrame["Q"].to_numpy()
        # data = self.dataFrame["data"]
        # I, Q = self.separate_iq_samples(data)
        data_complex = I + 1j*Q
        return data_complex
    
    def shuffle_in_groups_of_four(self, data, labels, rx, tx, seed=None):
        """Shuffle dataset by contiguous quadruplet blocks while preserving each block.

        The generator samples indices in steps of 4: [i, i+1, i+2, i+3].
        This function randomly permutes the order of those quadruplet blocks
        but keeps the items inside each block together and in-order.

        If the dataset length is not divisible by 4, the trailing samples
        are dropped so that all remaining samples form complete quadruplets.
        """
        # Use a local RNG when a seed is provided to avoid mutating global state
        rng = np.random.default_rng(seed) if seed is not None else None
        if data is None or labels is None:
            return data, labels, rx, tx

        num_samples = len(data)
        usable = (num_samples // 4) * 4
        if usable == 0:
            return data[:0], labels[:0], rx[:0], tx[:0]

        if usable != num_samples:
            # Drop trailing samples that don't form a complete quadruplet
            data = data[:usable]
            labels = labels[:usable]
            rx = rx[:usable]
            tx = tx[:usable]
        num_blocks = usable // 4
        block_order = (rng.permutation(num_blocks) if rng is not None else np.random.permutation(num_blocks))
        # Build the new index order by concatenating each block's four indices
        new_indices = []
        for b in block_order:
            base = 4 * b
            new_indices.extend([base, base + 1, base + 2, base + 3])

        new_indices = np.asarray(new_indices, dtype=int)
        data_shuffled = data[new_indices]
        labels_shuffled = labels[new_indices]
        rx_shuffled = rx[new_indices]
        tx_shuffled = tx[new_indices]
        return data_shuffled, labels_shuffled, rx_shuffled, tx_shuffled
    
    def load_data(self, shuffle=False, seed=None):
        label = self.dataFrame["channel"]
        # label = self.dataFrame["label"]
        label = label.astype(int)
        label = np.transpose(label)
        
        rx = self.dataFrame["rx"]
        tx = self.dataFrame["tx"]
        rx = rx.astype(int)
        tx = tx.astype(int)
        rx = np.transpose(rx)
        tx = np.transpose(tx)
        
        data = self.convert_to_complex()
        if shuffle:
            data, label, rx, tx = self.shuffle_in_groups_of_four(data, label, rx, tx, seed=seed)
        return data,label, rx, tx
    
    def plot_spectrogram(self,spectrogram, config_name):
        # In a figure with 4 subplots each showing a spectrogram for each quadruplet
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        labels = ["alice-bob", "alice-eve", "bob-alice", "bob-eve"]
        spec_index = 0
        for i in range(2):
            for j in range(2):
                axes[i,j].imshow(spectrogram[spec_index], aspect='auto', cmap='jet')
                axes[i,j].set_title(f'Spectrogram {labels[spec_index]}')
                axes[i,j].set_xlabel('Time')
                axes[i,j].set_ylabel('Frequency')
                spec_index += 1
        plt.tight_layout()
        plt.show()
        plt.savefig(f"spectrogram_{spectrogram.shape[0]}_{spectrogram.shape[1]}_{spectrogram.shape[2]}_{config_name}.png")
        
        # plt.figure(figsize=(10, 5))
        # plt.imshow(spectrogram, aspect='auto', cmap='jet')
        # plt.colorbar()
        # plt.title('Spectrogram')
        # plt.xlabel('Time')
        # plt.ylabel('Frequency')
        # plt.show()
        # plt.savefig(f"spectrogram_{spectrogram.shape[0]}_{spectrogram.shape[1]}_{spectrogram.shape[2]}.png")
        
    def plot_avg_signal_power(self, data, config_name, scenarios_info):
        # Get the average power for every sample in data
        # Plot avg power for each sample
        # Labels are distributed as quadruplets where each quadruplet is sample [i, i+1, i+2, i+3]
        # For each sample in the quadruplet lets assign a label [alice-bob, alice-eve, bob-alice, bob-eve]
        numScenarios = scenarios_info["numScenarios"]
        numSamplesPerScenario = scenarios_info["numSamplesPerScenario"]
        rx = scenarios_info["rx"]
        tx = scenarios_info["tx"]
        # avg_signal_power_arr = {"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]}
        scenario_avg_signal_power_arr = {"scenario-"+str(scenario):{"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]} for scenario in range(numScenarios)}
        for scenario in range(numScenarios):
            print("Scenario:", scenario)
            for sample in range(0, numSamplesPerScenario, 4):
                i = scenario*(numSamplesPerScenario) + sample
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["alice-bob"].append(np.mean(np.abs(data[i])**2))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["alice-eve"].append(np.mean(np.abs(data[i+1])**2))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["bob-alice"].append(np.mean(np.abs(data[i+2])**2))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["bob-eve"].append(np.mean(np.abs(data[i+3])**2))
        
        # plt.figure(figsize=(10, 5))
        # for label in avg_signal_power_arr.keys():
        #     plt.plot(avg_signal_power_arr[label], label=label)
        # plt.title('Average Signal Power')
        # plt.xlabel('Sample')
        # plt.ylabel('Power')
        # plt.legend()
        # plt.show()
        # plt.savefig(f"avg_signal_power_{data.shape[0]}_{config_name}.png")
        
        # Create a box plot where the x-axis is the scenario and the y-axis is the average signal power for each label
        # On every scenario plot a box plot for each label
        # Lets have a single plot where the x axis
        from matplotlib.patches import Patch
        label_names = ["alice-bob", "alice-eve", "bob-alice", "bob-eve"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        fig, ax = plt.subplots(figsize=(12, 6))
        positions_base = list(range(numScenarios))
        offsets = np.linspace(-0.3, 0.3, len(label_names))
        box_width = 0.18

        for l_idx, label_name in enumerate(label_names):
            positions = [p + offsets[l_idx] for p in positions_base]
            data_to_plot = [scenario_avg_signal_power_arr[f"scenario-{s}"][label_name] for s in range(numScenarios)]

            bp = ax.boxplot(
                data_to_plot,
                positions=positions,
                widths=box_width,
                patch_artist=True,
                manage_ticks=False,
                showfliers=False,
            )

            for box in bp["boxes"]:
                box.set_facecolor(colors[l_idx])
                box.set_alpha(0.7)
                box.set_edgecolor("black")
            for median in bp["medians"]:
                median.set_color("black")
                median.set_linewidth(1.5)

        ax.set_xticks(positions_base)
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Average Signal Power")
        ax.set_title(f"Average Signal Power per Scenario ({config_name})")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        legend_handles = [
            Patch(facecolor=colors[i], edgecolor="black", label=label_names[i], alpha=0.7)
            for i in range(len(label_names))
        ]
        ax.legend(handles=legend_handles, title="Label", loc="best")
        
        # Annotate RX number above the maximum point for each label within each scenario
        rx_arr = np.asarray(rx)
        label_offset_map = {"alice-bob": 0, "alice-eve": 1, "bob-alice": 2, "bob-eve": 3}
        global_max_val = None
        for l_idx, label_name in enumerate(label_names):
            offset_val = label_offset_map[label_name]
            for s in range(numScenarios):
                values = scenario_avg_signal_power_arr[f"scenario-{s}"][label_name]
                if len(values) == 0:
                    continue
                max_idx = int(np.argmax(values))
                max_val = float(values[max_idx])
                if (global_max_val is None) or (max_val > global_max_val):
                    global_max_val = max_val
                # Map back to sample index to fetch RX
                sample_index = s * numSamplesPerScenario + 4 * max_idx + offset_val
                rx_num = int(rx_arr[sample_index])
                x_pos = positions_base[s] + offsets[l_idx]
                ax.text(
                    x_pos,
                    max_val,
                    f"RX: {rx_num}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        # Ensure room for annotations globally
        if global_max_val is None:
            global_max_val = 1.0
        y_margin = 0.05 * (global_max_val if global_max_val != 0 else 1.0)
        ax.set_ylim(top=global_max_val + 2 * y_margin)

        plt.tight_layout()
        plt.savefig(f"avg_signal_power_{data.shape[0]}_{config_name}.png")
        plt.show()
    
class ChannelSpectrogram():
    def __init__(self,):
        pass
    
    def _normalization(self,data):
        ''' Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=np.complex64)
        
        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i]/rms
        
        return s_norm        

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
    
        return x_cropped

    def _gen_single_channel_spectrogram(self, sig, win_len=256, overlap=128):
        '''
        _gen_single_channel_ind_spectrogram converts the IQ samples to a channel
        independent spectrogram according to set window and overlap length.
        
        INPUT:
            SIG is the complex IQ samples.
            
            WIN_LEN is the window length used in STFT.
            
            OVERLAP is the overlap length used in STFT.
            
        RETURN:
            
            CHAN_IND_SPEC_AMP is the genereated channel independent spectrogram.
        '''
        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig, 
                                window='boxcar', 
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)
        
        # Take the logarithm of the magnitude.      
        chan_spec_amp = np.log10(np.abs(spec)**2)
        return chan_spec_amp
    
    def normalize_data(self, data):
        ''' Normalize the data with values between 0 and 1.'''
        data = data / np.max(data)
        return data
    
    def channel_spectrogram(self, data, FFTwindow=512):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        print("Data shape:", data.shape)
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        print("Data shape:", data.shape)
        # Normalize the IQ samples.
        data = self._normalization(data)
            
        # Calculate the size of channel independent spectrograms.
        win_len=FFTwindow # 128 | 256 | 512 --Smaller window will give better time resolution but worse freq. resolution and vice versa. 128 for N=8192 is the most balanced
        overlap=win_len/2
        
        num_sample = data.shape[0]
        # num_row = int(np.floor(win_len*0.4))
        num_row = int(win_len)
        num_column = int(np.ceil((data.shape[1]-win_len)/overlap + 1))
        data_channel_spec = np.zeros([num_sample, num_row, num_column, 1])
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_spec_amp = self._gen_single_channel_spectrogram(data[i],win_len, overlap)
            # chan_spec_amp = self._spec_crop(chan_spec_amp)
            chan_spec_amp = self.normalize_data(chan_spec_amp)
            data_channel_spec[i,:,:,0] = chan_spec_amp
            
        return data_channel_spec


    
if __name__ == "__main__":
    # Example usage
    dataset_name = "Key-Generation"
    # config_name = "Sinusoid-Powder-OTA-Dense-Nodes-123" #"Sinusoid-Powder-OTA-Lab"
    config_names= [
            # "Sinusoid-Powder-OTA-Dense-Nodes-123",
            # "Sinusoid-Powder-OTA-Dense-Nodes-125",
            # "Sinusoid-Powder-OTA-Dense-Nodes-132",
            # "Sinusoid-Powder-OTA-Dense-Nodes-435",
            "Sinusoid-Powder-OTA-Lab-Nodes-123",
            "Sinusoid-Powder-OTA-Lab-Nodes-145",
            "Sinusoid-Powder-OTA-Lab-Nodes-148",
            "Sinusoid-Powder-OTA-Lab-Nodes-243",
            "Sinusoid-Powder-OTA-Lab-Nodes-425",
            "Sinusoid-Powder-OTA-Lab-Nodes-428",
            "Sinusoid-Powder-OTA-Lab-Nodes-485",
            "Sinusoid-Powder-OTA-Lab-Nodes-578",
            "Sinusoid-Powder-OTA-Lab-Nodes-587",
            "Sinusoid-Powder-OTA-Lab-Nodes-841",
            "Sinusoid-Powder-OTA-Lab-Nodes-851",
            "Sinusoid-Powder-OTA-Lab-Nodes-854"
            ]
    
    # Lets concatenate the dataframes
    dataframes = []
    repo_name="CAAI-FAU"
    numScenarios = 0
    for idx, config_name in enumerate(config_names):
        if idx == 0:
            dataset_handler = DatasetHandler(dataset_name, config_name, repo_name)
        else:
            dataset_handler.add_dataset(dataset_name, config_name, repo_name)
        numScenarios += 1
    print("Number of scenarios:", numScenarios)
    numSamplesPerScenario = len(dataset_handler.dataFrame)//numScenarios
    print("Number of samples per scenario: ", numSamplesPerScenario)
    dataset_handler.get_dataframe_Info()
    data, labels, rx, tx = dataset_handler.load_data()
    group_name = "OTA-Lab"
    scenarios_info = {
        "numScenarios": numScenarios,
        "numSamplesPerScenario": numSamplesPerScenario,
        "rx": rx,
        "tx": tx
    }
    dataset_handler.plot_avg_signal_power(data, config_name = group_name, scenarios_info = scenarios_info)
    
    signal_processor = ChannelSpectrogram()
    # Print all unique labels, rx, tx
    print("Unique labels:", np.unique(labels))
    print("Unique rx:", np.unique(rx))
    print("Unique tx:", np.unique(tx))
        
    for FFTwindow in [128, 256, 512, 1024]:
        try:
            data_channel_spec = signal_processor.channel_spectrogram(data, FFTwindow=FFTwindow)
            print("Data channel spec shape FFTwindow=", FFTwindow, ":", data_channel_spec.shape)
            dataset_handler.plot_spectrogram(data_channel_spec, config_name = group_name)
        except Exception as e:
            print("Error in FFTwindow=", FFTwindow, ":", e)
        
    # data_channel_spec_128 = signal_processor.channel_spectrogram(data, FFTwindow=200)
    # print("Data channel spec shape FFTwindow=200:", data_channel_spec_128.shape)
    
    # plot_channel_spectrogram(data_channel_spec[0])
    # plot_channel_spectrogram(data_channel_spec[1])
    # plot_channel_spectrogram(data_channel_spec[2])
    # plot_channel_spectrogram(data_channel_spec[3])
    