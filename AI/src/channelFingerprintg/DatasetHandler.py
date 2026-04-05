import numpy as np
from datasets import Dataset, DatasetDict, DownloadMode
import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy import signal

# ------------------------------------------------------------
# Original code from gxhen repo https://github.com/gxhen/LoRa_RFFI.git
# ------------------------------------------------------------
# Dataset handler for the device fingerprinting dataset
# ------------------------------------------------------------

import numpy as np
import h5py
from numpy import sum,sqrt
from numpy.random import standard_normal, uniform

from scipy import signal

# In[]

def awgn(data, snr_range):
    
    pkt_num = data.shape[0]
    SNRdB = uniform(snr_range[0],snr_range[-1],pkt_num)
    for pktIdx in range(pkt_num):
        s = data[pktIdx]
        # SNRdB = uniform(snr_range[0],snr_range[-1])
        SNR_linear = 10**(SNRdB[pktIdx]/10)
        P= sum(abs(s)**2)/len(s)
        N0=P/SNR_linear
        n = sqrt(N0/2)*(standard_normal(len(s))+1j*standard_normal(len(s)))
        data[pktIdx] = s + n

    return data 

def _normalization(data):
    """Normalize each complex example to reduce power/offset leakage."""
    # return normalize_complex_samples(data, method=method, center=center)
    # Taking normalization from gxhen repo https://github.com/gxhen/LoRa_RFFI.git
    s_norm = np.zeros(data.shape, dtype=complex)
    
    for i in range(data.shape[0]):
    
        sig_amplitude = np.abs(data[i])
        rms = np.sqrt(np.mean(sig_amplitude**2))
        s_norm[i] = data[i]/rms
    
    return s_norm  

class LoadDataset():
    def __init__(self,):
        self.dataset_name = 'data'
        self.labelset_name = 'label'
        
    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1] 
        data_complex = np.zeros([num_row,round(num_col/2)],dtype=complex)
     
        data_complex = data[:,:round(num_col/2)] + 1j*data[:,round(num_col/2):] 
        return data_complex
    
    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.
        
        INPUT:
            FILE_PATH is the dataset path.
            
            DEV_RANGE specifies the loaded device range.
            
            PKT_RANGE specifies the loaded packets range.
            
        RETURN:
            DATA is the laoded complex IQ samples.
            
            LABLE is the true label of each received packet.
        '''
        
        f = h5py.File(file_path,'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1
        
        label_start = int(label[0]) + 1
        label_end = int(label[-1]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt/num_dev)
        
        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' + 
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')
        
        sample_index_list = []
        
        for dev_idx in dev_range:
            sample_index_dev = np.where(label==dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
    
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)
        
        label = label[sample_index_list]
          
        f.close()
        return data,label



class ChannelIndSpectrogram():
    def __init__(self,):
        pass

    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
    
        return x_cropped


    def _gen_single_channel_ind_spectrogram(self, sig, win_len=256, overlap=128):
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
        
        # Generate channel independent spectrogram.
        chan_ind_spec = spec[:,1:]/spec[:,:-1]    
        
        # Take the logarithm of the magnitude.      
        chan_ind_spec_amp = np.log10(np.abs(chan_ind_spec)**2)
                  
        return chan_ind_spec_amp
    


    def channel_ind_spectrogram(self, data):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        
        # Normalize the IQ samples.
        data = _normalization(data)
        
        # Calculate the size of channel independent spectrograms.
        num_sample = data.shape[0]
        num_row = int(256*0.4)
        num_column = int(np.floor((data.shape[1]-256)/128 + 1) - 1)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
                   
            chan_ind_spec_amp = self._gen_single_channel_ind_spectrogram(data[i])
            chan_ind_spec_amp = self._spec_crop(chan_ind_spec_amp)
            data_channel_ind_spec[i,:,:,0] = chan_ind_spec_amp
            
        return data_channel_ind_spec
    
# ------------------------------------------------------------
# End of original code from gxhen repo https://github.com/gxhen/LoRa_RFFI.git
# ------------------------------------------------------------

# ------------------------------------------------------------
# Custom dataset handler for the channel fingerprinting dataset
# ------------------------------------------------------------

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

    def evaluate_dataset_diagnostics(
        self,
        dataset_label,
        output_dir,
        samples_per_scenario=400,
        show_plot=False,
        iq_probe_index=None,
    ):
        """
        Generate diagnostic plots/metrics for one dataset configuration.

        This helps explain why some scenarios produce better/worse key quality by
        comparing link behaviors:
          - Alice-Bob (AB)
          - Alice-Eve (AE)
          - Bob-Alice (BA)
          - Bob-Eve (BE)
        """
        os.makedirs(output_dir, exist_ok=True)
        data, labels, rx, tx = self.load_data(shuffle=False)
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])

        usable = (data.shape[0] // 4) * 4
        if usable == 0:
            raise ValueError("Dataset has fewer than 4 samples; cannot form AB/AE/BA/BE quadruplets.")
        if usable != data.shape[0]:
            data = data[:usable]
            labels = labels[:usable]
            rx = rx[:usable]
            tx = tx[:usable]

        ab = data[0::4]
        ae = data[1::4]
        ba = data[2::4]
        be = data[3::4]
        link_data = {
            "alice-bob": ab,
            "alice-eve": ae,
            "bob-alice": ba,
            "bob-eve": be,
        }

        # ---- Plot 1: IQ samples for one probe per link ----
        n_probes = ab.shape[0]
        # By default, visualize a representative probe from the middle.
        if iq_probe_index is None:
            probe_idx = n_probes // 2
        else:
            probe_idx = int(np.clip(iq_probe_index, 0, n_probes - 1))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        for i, (name, arr) in enumerate(link_data.items()):
            sig = arr[probe_idx]
            axes[i].plot(sig.real, label="I", linewidth=1.2)
            axes[i].plot(sig.imag, label="Q", linewidth=1.2)
            axes[i].set_title(f"{name} - probe {probe_idx}")
            axes[i].set_xlabel("Sample")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(alpha=0.3)
            axes[i].legend()
        plt.tight_layout()
        iq_plot_path = os.path.join(output_dir, f"{dataset_label}_diag_iq_links.png")
        plt.savefig(iq_plot_path, dpi=180)
        if show_plot:
            plt.show()
        plt.close(fig)

        # ---- Probe power helpers ----
        def avg_probe_power(x):
            return np.mean(np.abs(x) ** 2, axis=1)

        power = {name: avg_probe_power(arr) for name, arr in link_data.items()}

        # ---- Plot 2: Per-probe average power by link ----
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, p in power.items():
            ax.plot(p, label=name, linewidth=1.2)
        ax.set_title("Average probe power by link")
        ax.set_xlabel("Probe index")
        ax.set_ylabel("Average power")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        power_curve_path = os.path.join(output_dir, f"{dataset_label}_diag_probe_power_curve.png")
        plt.savefig(power_curve_path, dpi=180)
        if show_plot:
            plt.show()
        plt.close(fig)

        # ---- Plot 3: Power distribution per link ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(
            [power["alice-bob"], power["alice-eve"], power["bob-alice"], power["bob-eve"]],
            labels=["AB", "AE", "BA", "BE"],
            showfliers=False,
        )
        ax.set_title("Probe power distribution by link")
        ax.set_ylabel("Average power")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        power_box_path = os.path.join(output_dir, f"{dataset_label}_diag_probe_power_box.png")
        plt.savefig(power_box_path, dpi=180)
        if show_plot:
            plt.show()
        plt.close(fig)

        # ---- Plot 4: Scenario-wise mean power (if enough probes) ----
        scenario_plot_path = None
        probes_per_scenario = max(1, samples_per_scenario // 4)
        num_scenarios = n_probes // probes_per_scenario
        if num_scenarios > 0:
            fig, ax = plt.subplots(figsize=(10, 5))
            for name, p in power.items():
                scen_means = []
                for s in range(num_scenarios):
                    start = s * probes_per_scenario
                    stop = start + probes_per_scenario
                    scen_means.append(float(np.mean(p[start:stop])))
                ax.plot(scen_means, marker="o", label=name)
            ax.set_title("Scenario-wise mean probe power by link")
            ax.set_xlabel("Scenario index")
            ax.set_ylabel("Mean probe power")
            ax.grid(alpha=0.3)
            ax.legend()
            plt.tight_layout()
            scenario_plot_path = os.path.join(output_dir, f"{dataset_label}_diag_scenario_power.png")
            plt.savefig(scenario_plot_path, dpi=180)
            if show_plot:
                plt.show()
            plt.close(fig)

        # ---- Pairwise link similarity (complex correlation magnitude) ----
        def link_similarity(x, y, eps=1e-12):
            # Returns one similarity value per probe in [0, 1].
            num = np.abs(np.sum(x * np.conj(y), axis=1))
            den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + eps
            return num / den

        similarities = {
            "AB-BA": link_similarity(ab, ba),
            "AB-AE": link_similarity(ab, ae),
            "BA-BE": link_similarity(ba, be),
            "AB-BE": link_similarity(ab, be),
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(
            [similarities["AB-BA"], similarities["AB-AE"], similarities["BA-BE"], similarities["AB-BE"]],
            labels=["AB-BA", "AB-AE", "BA-BE", "AB-BE"],
            showfliers=False,
        )
        ax.set_title("Pairwise link similarity distribution")
        ax.set_ylabel("Similarity (|corr|)")
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        similarity_plot_path = os.path.join(output_dir, f"{dataset_label}_diag_link_similarity.png")
        plt.savefig(similarity_plot_path, dpi=180)
        if show_plot:
            plt.show()
        plt.close(fig)

        summary = {
            "dataset_label": dataset_label,
            "num_samples_total": int(data.shape[0]),
            "num_probes_per_link": int(n_probes),
            "mean_power": {k: float(np.mean(v)) for k, v in power.items()},
            "std_power": {k: float(np.std(v)) for k, v in power.items()},
            "mean_similarity": {k: float(np.mean(v)) for k, v in similarities.items()},
            "plots": {
                "iq_links": iq_plot_path,
                "probe_power_curve": power_curve_path,
                "probe_power_box": power_box_path,
                "scenario_power": scenario_plot_path,
                "link_similarity": similarity_plot_path,
            },
        }

        print("Dataset diagnostics summary:", summary)
        return summary
    
    def plot_time_series(self, data, config_name, indexes_to_plot=None):
        # In a figure with 4 subplots each showing a time series for each quadruplet
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        print("Data shape:", data.shape)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        labels = ["alice-bob", "alice-eve", "bob-alice", "bob-eve"]
        time_series_starting_index = 240
        signal_index = 0
        for i in range(2):
            for j in range(2):
                sample_index = signal_index+time_series_starting_index if indexes_to_plot is None else indexes_to_plot[signal_index]
                realSample = data[sample_index].real
                imagSample = data[sample_index].imag
                axes[i,j].plot(realSample, label='Real')
                axes[i,j].plot(imagSample, label='Imaginary')
                axes[i,j].set_title(f'Time Series {labels[signal_index]} at index {sample_index}')
                axes[i,j].set_xlabel('Sample index')
                axes[i,j].set_ylabel('Amplitude')
                axes[i,j].legend(loc='upper right')
                signal_index += 1
        plt.tight_layout()
        plt.show()
        plt.savefig(f"time_series_{config_name}.png")
    
    def complex_to_polar(self, data):
        I = data.real
        Q = data.imag
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        return I, Q,amplitude, phase
    
    def plot_polar_coordinates(self, data, config_name, indexes_to_plot=None):
        # In a figure with 4 subplots each showing a polar coordinates for each quadruplet
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        print("Data shape:", data.shape)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
        labels = ["alice-bob", "alice-eve", "bob-alice", "bob-eve"]
        time_series_starting_index = 240
        signal_index = 0
        for i in range(2):
            for j in range(2):
                sample_index = signal_index+time_series_starting_index if indexes_to_plot is None else indexes_to_plot[signal_index]
                
                I, Q, amplitude, phase = self.complex_to_polar(data[sample_index])
                # Use unit radius for a clean circle visualization unless disabled
                r = amplitude
                x = r * np.cos(phase)
                y = r * np.sin(phase)
                axes[i,j].scatter(x, y, s=5, alpha=0.7)
                max_r = float(np.max(amplitude)) if amplitude.size else 1.0
                axes[i,j].set_xlim(-max_r * 1.05, max_r * 1.05)
                axes[i,j].set_ylim(-max_r * 1.05, max_r * 1.05)
                axes[i,j].set_aspect('equal', adjustable='box')
                axes[i,j].set_title(f'Polar Coordinates {labels[signal_index]} at index {sample_index}')
                axes[i,j].set_xlabel('I')
                axes[i,j].set_ylabel('Q')
                signal_index += 1
        plt.tight_layout()
        plt.savefig(f"polar_coordinates_{config_name}.png")
        plt.show()
    
    def plot_spectrogram(self,spectrogram, config_name, indexes_to_plot=None):
        # In a figure with 4 subplots each showing a spectrogram for each quadruplet
        # spectrogram = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in spectrogram])
        print("Spectrogram shape:", spectrogram.shape)
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
        labels = ["alice-bob", "alice-eve", "bob-alice", "bob-eve"]
        spec_index = 0
        # show spectogram dimensions
        spectrogram_starting_index = 200
        print("Spectrogram dimensions:", spectrogram.shape)
        
        # Determine selected indices (4 total)
        selected_indices = []
        for k in range(4):
            idx = (k + spectrogram_starting_index) if indexes_to_plot is None else indexes_to_plot[k]
            selected_indices.append(int(idx))

        # Compute unified color scale across selected spectrograms
        def to_2d(img):
            return img[:, :, 0] if (img.ndim == 3 and img.shape[-1] == 1) else img
        vals_min = []
        vals_max = []
        for idx in selected_indices:
            arr2d = to_2d(np.asarray(spectrogram[idx]))
            vals_min.append(np.min(arr2d))
            vals_max.append(np.max(arr2d))
        vmin = float(np.min(vals_min))
        vmax = float(np.max(vals_max))
        unified_img = None
        for i in range(4):
            # for j in range(4):
            sample_index = selected_indices[spec_index]
            arr = np.asarray(spectrogram[sample_index])
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[:, :, 0]
            print("Spectrogram dimensions:", arr.shape)
            img = axes[i].imshow(arr, aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
            axes[i].set_title(f'{labels[spec_index]}')
            axes[i].set_xlabel('M')
            axes[i].set_ylabel('N')
            # Ensure x and y axis show the starting and ending indices of the spectrogra showing every value
            axes[i].set_xlim(0, arr.shape[1])
            axes[i].set_ylim(0, arr.shape[0])
            # Add start/end and >=10 intermediate tick marks on both axes
            height = arr.shape[0]
            width = arr.shape[1]
            num_ticks = 12 if width >= 12 else max(width, 2)
            xticks = np.linspace(0, width - 1, num=num_ticks, dtype=int)
            axes[i].set_xticks(xticks)
            axes[i].set_xticklabels([str(int(x)) for x in xticks])
            num_ticks_y = 12 if height >= 12 else max(height, 2)
            yticks = np.linspace(0, height - 1, num=num_ticks_y, dtype=int)
            axes[i].set_yticks(yticks)
            axes[i].set_yticklabels([str(int(y)) for y in yticks])
            unified_img = img
            spec_index += 1
        # Add a single unified colorbar for all subplots (outside on the right)
        if unified_img is not None:
            # Reserve space on the right for the colorbar
            plt.tight_layout(rect=[0.0, 0.0, 0.92, 1.0])
            cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
            fig.colorbar(unified_img, cax=cbar_ax)
        else:
            plt.tight_layout()
        plt.show()
        plt.savefig(f"spectrogram_{spectrogram.shape[0]}_{spectrogram.shape[1]}_{spectrogram.shape[2]}_{config_name}.png")

        
    def plot_avg_signal_power(self, data, config_name, scenarios_info):
        # Get the average power for every sample in data
        # Plot avg power for each sample
        # Labels are distributed as quadruplets where each quadruplet is sample [i, i+1, i+2, i+3]
        # For each sample in the quadruplet lets assign a label [alice-bob, alice-eve, bob-alice, bob-eve]
        def avg_signal_power(data):
            return np.mean(np.abs(data)**2)
        numScenarios = scenarios_info["numScenarios"]
        numSamplesPerScenario = scenarios_info["numSamplesPerScenario"]
        rx = scenarios_info["rx"]
        tx = scenarios_info["tx"]
        # avg_signal_power_arr = {"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]}
        scenario_avg_signal_power_arr = {"scenario-"+str(scenario):{"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]} for scenario in range(numScenarios)}
        scenario_avg_signal_power_idx = {"scenario-"+str(scenario):{"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]} for scenario in range(numScenarios)}
        for scenario in range(numScenarios):
            print("Scenario:", scenario)
            for sample in range(0, numSamplesPerScenario, 4):
                i = scenario*(numSamplesPerScenario) + sample
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["alice-bob"].append(avg_signal_power(data[i]))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["alice-eve"].append(avg_signal_power(data[i+1]))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["bob-alice"].append(avg_signal_power(data[i+2]))
                scenario_avg_signal_power_arr["scenario-"+str(scenario)]["bob-eve"].append(avg_signal_power(data[i+3]))
                scenario_avg_signal_power_idx["scenario-"+str(scenario)]["alice-bob"].append(i)
                scenario_avg_signal_power_idx["scenario-"+str(scenario)]["alice-eve"].append(i+1)
                scenario_avg_signal_power_idx["scenario-"+str(scenario)]["bob-alice"].append(i+2)
                scenario_avg_signal_power_idx["scenario-"+str(scenario)]["bob-eve"].append(i+3)
        
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
        
        # ax.set_ylim(top=global_max_val + 2 * y_margin)
        # set Y axis to start from the lowest value in the data to the global maximum value
        global_min_val = np.min([np.min(scenario_avg_signal_power_arr[f"scenario-{s}"][label_name]) for s in range(numScenarios) for label_name in label_names])
        global_max_val = np.max([np.max(scenario_avg_signal_power_arr[f"scenario-{s}"][label_name]) for s in range(numScenarios) for label_name in label_names])
        print("global_max_val:", global_max_val)
        print("global_min_val:", global_min_val)
        print("y_margin:", y_margin)
        ax.set_ylim(bottom=global_min_val, top=global_max_val)
        ax.set_title(f"Average Signal Power per Scenario ({config_name})")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.savefig(f"avg_signal_power_{data.shape[0]}_{config_name}.png")
        plt.show()
        
        # Get the the index for the samples at the average signal power maximum for each label
        scenario_avg_signal_power_max_idx = {"scenario-"+str(scenario):{"alice-bob":[], "alice-eve":[], "bob-alice":[], "bob-eve":[]} for scenario in range(numScenarios)}
        for scenario in range(numScenarios):
            for label in label_names:
                scenario_avg_signal_power_max_idx["scenario-"+str(scenario)][label] = scenario_avg_signal_power_idx["scenario-"+str(scenario)][label][np.argmax(scenario_avg_signal_power_arr["scenario-"+str(scenario)][label])]
        print("Scenario avg signal power max idx:", scenario_avg_signal_power_max_idx)
        return scenario_avg_signal_power_max_idx
    
class ChannelSpectrogram():
    def __init__(self,):
        pass
    
    def _spec_crop(self, x):
        '''Crop the generated channel independent spectrogram.'''
        num_row = x.shape[0]
        x_cropped = x[round(num_row*0.3):round(num_row*0.7)]
        # x_cropped = x[round(num_row*0.1):round(num_row*0.9)]
    
        return x_cropped

    def _gen_single_channel_spectrogram(self, sig, win_len=256, overlap=128, processing="magnitude"):
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
        # Clamp STFT parameters for short signals (e.g., WiFi chan_est probes with 128 samples).
        sig_len = int(np.asarray(sig).shape[0])
        win_len = int(min(max(2, int(win_len)), max(2, sig_len)))
        overlap = int(min(max(0, int(overlap)), win_len - 1))

        # Short-time Fourier transform (STFT).
        f, t, spec = signal.stft(sig,
                                # fs=1000000,
                                window='boxcar', 
                                nperseg= win_len, 
                                noverlap= overlap, 
                                nfft= win_len,
                                return_onesided=False, 
                                padded = False, 
                                boundary = None)
        
        # FFT shift to adjust the central frequency.
        spec = np.fft.fftshift(spec, axes=0)
        
        mode = str(processing).strip().lower()
        # Take the logarithm of the magnitude.
        if mode == "magnitude":
            chan_spec_amp = np.log10(np.abs(spec) ** 2).astype(np.float32)
        elif mode == "complex":
            # Two-channel tensor [real, imag], analogous to multi-channel images.
            chan_spec_amp = np.stack([spec.real, spec.imag], axis=-1).astype(np.float32)
        elif mode in ("magnitude_phase", "mag_phase"):
            # chan_spec_log_mag = np.log10(np.abs(spec) ** 2).astype(np.float32)
            # Take the amplitude of the complex numbers
            chan_spec_amplitude = np.sqrt(spec.real**2 + spec.imag**2).astype(np.float32)
            # np.angle returns the arctan(b/a) where b is the imaginary part and a is the real part
            chan_spec_phase = np.angle(spec).astype(np.float32)
            # Two-channel tensor [magnitude, phase].
            chan_spec_amp = np.stack([chan_spec_amplitude, chan_spec_phase], axis=-1)
        else:
            raise ValueError(f"Invalid processing: {processing}")
        return chan_spec_amp
    
    def normalize_data(self, data):
        ''' Normalize the data with values between 0 and 1.'''
        min_val = np.min(data)
        max_val = np.max(data)
        if (max_val - min_val) == 0:  # Handle cases where all values are the same
            return np.zeros_like(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data
        
    def channel_spectrogram(self, data, FFTwindow=512, processing="magnitude", normalize=True, crop=True):
        '''
        channel_ind_spectrogram converts IQ samples to channel independent 
        spectrograms.
        
        INPUT:
            DATA is the IQ samples.
            
        RETURN:
            DATA_CHANNEL_IND_SPEC is channel independent spectrograms.
        '''
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        # Normalize complex IQ samples to reduce absolute-power shortcuts.
        # Normalizing taking into account entire batch of samples
        if normalize:
            data = _normalization(data)
            
        # Calculate the size of channel independent spectrograms.
        win_len=FFTwindow # 128 | 256 | 512 --Smaller window will give better time resolution but worse freq. resolution and vice versa. 128 for N=8192 is the most balanced
        overlap = int(win_len / 2)
        
        num_sample = data.shape[0]
        # Probe first sample to determine output dimensions/channels for the selected mode.
        first_spec = self._gen_single_channel_spectrogram(
            data[0], win_len, overlap, processing=processing
        )
        if crop:
            first_spec = self._spec_crop(first_spec)
        # Getting spectrogram dimensions
        if first_spec.ndim == 2:
            num_row, num_column = first_spec.shape
            num_channels = 1
        elif first_spec.ndim == 3:
            num_row, num_column, num_channels = first_spec.shape
        else:
            raise ValueError(
                f"Unsupported spectrogram tensor rank {first_spec.ndim} for processing={processing}"
            )
        data_channel_spec = np.zeros([num_sample, num_row, num_column, num_channels], dtype=np.float32)
        
        # Convert each packet (IQ samples) to a channel independent spectrogram.
        for i in range(num_sample):
            chan_spec_amp = self._gen_single_channel_spectrogram(
                data[i], win_len, overlap, processing=processing
            )
            if crop:
                chan_spec_amp = self._spec_crop(chan_spec_amp)
            if chan_spec_amp.ndim == 2:
                data_channel_spec[i, :, :, 0] = chan_spec_amp
            else:
                data_channel_spec[i, :, :, :] = chan_spec_amp
            
        return data_channel_spec

class ChannelIQ():
    def __init__(self,):
        pass
    
    def channel_iq(self, data, normalize=True, normalization_method="rms", center=True, rnn_format=False):
        '''
        channel_iq converts IQ samples to channel independent IQ samples.

        When rnn_format=True, output shape is (M, 1, N, 2) so that an RNN
        (which permutes to time-first) sees N time steps of [I, Q] features.
        '''
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        if normalize:
            data = _normalization(data)
        M, N = data.shape[0], data.shape[1]
        if rnn_format:
            data_iq = np.empty((M, 1, N, 2), dtype=np.float32)
            data_iq[:, 0, :, 0] = data.real.astype(np.float32)
            data_iq[:, 0, :, 1] = data.imag.astype(np.float32)
            print("Data IQ shape (M, 1, N, 2) [RNN-format]:", data_iq.shape)
        else:
            data_iq = np.empty((M, N, 2, 1), dtype=np.float32)
            data_iq[:, :, 0, 0] = data.real.astype(np.float32)
            data_iq[:, :, 1, 0] = data.imag.astype(np.float32)
            print("Data IQ shape (M, N, 2, 1):", data_iq.shape)
        return data_iq
    
class ChannelPolar():
    def __init__(self,):
        pass
    
    def complex_to_polar(self, data):
        I = data.real
        Q = data.imag
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        return amplitude, phase
    
    def channel_polar(self, data, normalize=True, normalization_method="rms", center=True):
        '''
        channel_polar converts IQ samples to channel independent polar samples.
        '''
        data = np.stack([np.asarray(pkt, dtype=np.complex64) for pkt in data])
        if normalize:
            data = _normalization(data)
        amplitude, phase = self.complex_to_polar(data)
        # Create (M, N, 2, 1) array: feature axis [Amplitude, Phase], trailing channel axis = 1
        M, N = amplitude.shape[0], amplitude.shape[1]
        data_polar = np.empty((M, N, 2, 1), dtype=np.float32)
        data_polar[:, :, 0, 0] = amplitude.astype(np.float32)
        data_polar[:, :, 1, 0] = phase.astype(np.float32)
        print("Data Polar shape (M, N, 2, 1):", data_polar.shape)
        return data_polar
    
def build_group_names(config_names):
    """Build concatenated Dense/Lab group names from config names.

    Input examples per item:
      "Sinusoid-Powder-OTA-Dense-Nodes-435"
      "Sinusoid-Powder-OTA-Lab-Nodes-123"

    Returns (dense_config_names, lab_config_names), e.g.:
      ("OTA-Dense-456-123-323", "OTA-Lab-432-123-323")
    """
    dense_ids = []
    lab_ids = []
    for name in config_names:
        if "OTA-Dense-Nodes-" in name:
            suffix = name.split("OTA-Dense-Nodes-", 1)[1]
            dense_ids.append(suffix.split("-")[0])
        if "OTA-Lab-Nodes-" in name:
            suffix = name.split("OTA-Lab-Nodes-", 1)[1]
            lab_ids.append(suffix.split("-")[0])

    # Deduplicate while preserving original order
    dense_ids = list(dict.fromkeys(dense_ids))
    lab_ids = list(dict.fromkeys(lab_ids))

    dense_config_names = f"OTA-Dense-{'-'.join(dense_ids)}" if dense_ids else None
    lab_config_names = f"OTA-Lab-{'-'.join(lab_ids)}" if lab_ids else None
    return dense_config_names, lab_config_names




SOURCE_OTA_LAB = 0
SOURCE_OTA_DENSE = 1
SOURCE_SIONNA = 2


def load_cached_dataset(
    cache_path,
    source_filter=None,
    node_ids_filter=None,
):
    """Load pre-built HDF5 dataset cache.

    Parameters
    ----------
    cache_path : str
        Path to the HDF5 file written by preload_dataset.py.
    source_filter : list[int] | None
        Keep only samples whose ``source`` value is in this list.
        0 = OTA-Lab, 1 = OTA-Dense, 2 = Sionna.  None keeps everything.
    node_ids_filter : list[list[int]] | None
        Keep only samples matching specific [alice, bob, eve] triples.
        Example: [[1,2,3], [5,8,7]] keeps scenarios with those exact nodes.
        None keeps everything.

    Returns
    -------
    dict with keys:
        data        : np.ndarray complex64 (N, num_samples)
        labels      : np.ndarray int32     (N,)
        rx          : np.ndarray int32     (N,)
        tx          : np.ndarray int32     (N,)
        source      : np.ndarray uint8     (N,)
        alice       : np.ndarray int32     (N,)
        bob         : np.ndarray int32     (N,)
        eve         : np.ndarray int32     (N,)
        scenario_index : np.ndarray int32  (N,)
        attrs       : dict of file-level attributes
    """
    import h5py as _h5py
    import json as _json

    with _h5py.File(cache_path, "r") as f:
        iq = f["iq_data"][:]
        labels = f["labels"][:]
        rx = f["rx"][:]
        tx = f["tx"][:]
        source = f["source"][:]
        alice = f["alice"][:]
        bob = f["bob"][:]
        eve = f["eve"][:]
        scenario_index = f["scenario_index"][:]
        attrs = dict(f.attrs)
        if "source_names" in attrs:
            attrs["source_names"] = _json.loads(attrs["source_names"])

    data = iq[:, :, 0] + 1j * iq[:, :, 1]
    data = data.astype(np.complex64)
    del iq

    mask = np.ones(len(data), dtype=bool)

    if source_filter is not None:
        source_set = set(int(s) for s in source_filter)
        mask &= np.isin(source, list(source_set))

    if node_ids_filter is not None:
        node_mask = np.zeros(len(data), dtype=bool)
        for nids in node_ids_filter:
            node_mask |= (alice == nids[0]) & (bob == nids[1]) & (eve == nids[2])
        mask &= node_mask

    if not mask.all():
        idx = np.where(mask)[0]
        data = data[idx]
        labels = labels[idx]
        rx = rx[idx]
        tx = tx[idx]
        source = source[idx]
        alice = alice[idx]
        bob = bob[idx]
        eve = eve[idx]
        scenario_index = scenario_index[idx]

    print(f"Loaded {len(data)} samples from cache: {cache_path}")
    return {
        "data": data,
        "labels": labels,
        "rx": rx,
        "tx": tx,
        "source": source,
        "alice": alice,
        "bob": bob,
        "eve": eve,
        "scenario_index": scenario_index,
        "attrs": attrs,
    }


if __name__ == "__main__":
    dataset = DatasetHandler("Key-Generation", "Sinusoid-Powder-OTA-Dense-Nodes-123", "CAAI-FAU")
    summary_123 = dataset.evaluate_dataset_diagnostics(
        dataset_label="OTA-Dense-123",
        output_dir="/home/Research/POWDER/Results/DatasetDiagnostics",
        samples_per_scenario=400,
        show_plot=False,
    )
    dataset2 = DatasetHandler("Key-Generation", "Sinusoid-Powder-OTA-Dense-Nodes-132", "CAAI-FAU")
    summary_132 = dataset2.evaluate_dataset_diagnostics(
        dataset_label="OTA-Dense-132",
        output_dir="/home/Research/POWDER/Results/DatasetDiagnostics",
        samples_per_scenario=400,
        show_plot=False,
    )
    exit()
    # Example usage
    dataset_name = "Key-Generation"
    # config_name = "Sinusoid-Powder-OTA-Dense-Nodes-123" #"Sinusoid-Powder-OTA-Lab"
    config_names= [
            # "Sinusoid-Powder-OTA-Dense-Nodes-123",
            # "Sinusoid-Powder-OTA-Dense-Nodes-125",
            "Sinusoid-Powder-OTA-Dense-Nodes-132",
            # "Sinusoid-Powder-OTA-Dense-Nodes-435",
            # "Sinusoid-Powder-OTA-Lab-Nodes-123",
            # "Sinusoid-Powder-OTA-Lab-Nodes-145",
            # "Sinusoid-Powder-OTA-Lab-Nodes-148",
            # "Sinusoid-Powder-OTA-Lab-Nodes-243",
            # "Sinusoid-Powder-OTA-Lab-Nodes-425",
            # "Sinusoid-Powder-OTA-Lab-Nodes-428",
            # "Sinusoid-Powder-OTA-Lab-Nodes-485",
            # "Sinusoid-Powder-OTA-Lab-Nodes-578",
            # "Sinusoid-Powder-OTA-Lab-Nodes-587",
            # "Sinusoid-Powder-OTA-Lab-Nodes-841",
            # "Sinusoid-Powder-OTA-Lab-Nodes-851",
            # "Sinusoid-Powder-OTA-Lab-Nodes-854" 
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
    signal_processor = ChannelSpectrogram()
    data = _normalization(data)
    dense_config_names, lab_config_names = build_group_names(config_names)
    print("Dense config names:", dense_config_names)
    print("Lab config names:", lab_config_names)
    group_name = dense_config_names + "-" + lab_config_names if dense_config_names and lab_config_names else dense_config_names or lab_config_names
    print("Group name:", group_name)
    # exit()
    scenarios_info = {
        "numScenarios": numScenarios,
        "numSamplesPerScenario": numSamplesPerScenario,
        "rx": rx,
        "tx": tx
    }
    scenarios_indexes_avg_signal_power = dataset_handler.plot_avg_signal_power(data, config_name = group_name, scenarios_info = scenarios_info)
    
    
    # Print all unique labels, rx, tx
    print("Unique labels:", np.unique(labels))
    print("Unique rx:", np.unique(rx))
    print("Unique tx:", np.unique(tx))
    indexes_to_plot = [index for index in scenarios_indexes_avg_signal_power["scenario-0"].values()]
    print("Indexes to plot:", indexes_to_plot)
    indexes_to_plot = None
    # exit()
    dataset_handler.plot_time_series(data, config_name = group_name, indexes_to_plot = indexes_to_plot)
    dataset_handler.plot_polar_coordinates(data, config_name = group_name, indexes_to_plot = indexes_to_plot)
    
    for FFTwindow in [256]:
        # try:
        data_channel_spec = signal_processor.channel_spectrogram(data, FFTwindow=FFTwindow)
        print("Data channel spec shape FFTwindow=", FFTwindow, ":", data_channel_spec.shape)
        dataset_handler.plot_spectrogram(data_channel_spec, config_name = group_name, indexes_to_plot = indexes_to_plot)
        # except Exception as e:
        #     print("Error in FFTwindow=", FFTwindow, ":", e)
        
    # data_channel_spec_128 = signal_processor.channel_spectrogram(data, FFTwindow=200)
    # print("Data channel spec shape FFTwindow=200:", data_channel_spec_128.shape)
    
    # plot_channel_spectrogram(data_channel_spec[0])
    # plot_channel_spectrogram(data_channel_spec[1])
    # plot_channel_spectrogram(data_channel_spec[2])
    # plot_channel_spectrogram(data_channel_spec[3])
    