import h5py
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
import huggingface_hub as hf

class DatasetGenerator():
    def __init__(self, fileName):
        self.fileName = fileName
        self.data = None
        self.dataFrame = None
        self.read_hdf5_file()
        self.generate_dataframe_from_hdf5()

    # Save dataframe to huggingface dataset
    def saveDataFrame(self, dataset_name, config_name, repo_name="CAAI-FAU"):
        username = hf.whoami()['name']
        print("Pushing dataset to Hugging Face hub...")
        # Create a new dataset
        print("Creating new dataset...")
        # Convert to HF Dataset
        hf_dataset = Dataset.from_pandas(self.dataFrame)
        # Push to Hugging Face hub
        repo_name = username if repo_name == "" else repo_name
        hf_dataset.push_to_hub(repo_name+"/"+dataset_name, private=True, config_name=config_name)
        print("Dataset pushed to Hugging Face hub successfully.")

    # read hdf5 file
    def read_hdf5_file(self):
        data = {}
        with h5py.File(self.fileName, 'r') as f:
            # Assuming the dataset is named 'dataset'
            # Read all key from the file
            keys = list(f.keys())
            for key in keys:
                data[key] = f[key][:]
        self.data = data

    def generate_dataframe_from_hdf5(self):
        for key in self.data.keys(): # Read all keys in dictionary
            if self.data[key].shape[0] == 1: # If first dimension shape is 1, remove it
                self.data[key] = self.data[key][0]
            if len(self.data[key].shape) > 1: # If data shape is 2D, make it a list
                self.data[key] = list(self.data[key])
        self.dataFrame = pd.DataFrame(self.data)

    def separate_iq_samples(self, data):
        # I is the second half of the samples
        I = data[len(data)//2:]
        # Q is the first half of the samples
        Q = data[:len(data)//2]
        return I, Q

    def plot_iq_samples(self, I,Q):
        # Plot IQ samples
        plt.plot(I, label='I')
        plt.plot(Q, label='Q')
        plt.title('IQ Samples')
        plt.xlabel('Sample Number')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def plot_quadruplet_samples(self, ids):
        # Plot quadrupole samples
        dataInstance = self.dataFrame[self.dataFrame['ids'] == ids]
        for index, row in dataInstance.iterrows():
            data = row['data']
            I, Q = self.separate_iq_samples(data)
            plt.plot(I, label='I')
            plt.plot(Q, label='Q')
            name = "Eve" if row["instance"] % 2 == 0 else "Alice" if row["instance"] == 1 else "Bob"
            plt.title(f'IQ Samples. ID: {ids}, label: {row["label"]}, instance: {row["instance"]} for {name}')
            plt.xlabel('Sample Number')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()    
    
if __name__ == "__main__":
    # Example usage
    saveDataFrame = False
    names = ['Alice', 'Bob', 'Eve']
    nodeConfigs = {
        "IDs":[
            [1,2,3],
            [2,4,3],
            [4,2,8],
            [4,2,5],
            [5,6,7],
            # [5,7,6],
            # [5,8,1],
            # [5,8,3]
        ],
        "Timestamps":[
            1747672681,
            1747673451,
            1747674226,
            1747675004,
            1747670000,
            # 1747670000,
            # 1747670000,
            # 1747670000,
        ]
    }
    # nodeIDs = [3,4,5]
    # timestamp = 1746944796
    numProbes = 100
    signalType = "sinusoid"
    folder = "/Users/josea/Workspaces/PowderKeyGen/"
    for nodeIDs, timestamp in zip(nodeConfigs["IDs"], nodeConfigs["Timestamps"]):
        file = folder + "Dataset_Channels_"+signalType+"_"+str(numProbes)+"_"+"".join(str(node) for node in nodeIDs)+"_"+str(timestamp)+".hdf5"
        print("Reading file: ", file)
        # Read the hdf5 file
        dataset = DatasetGenerator(file)
        # Print the dataframe info
        # dataset.get_dataframe_Info()
        # Plot the IQ samples
        # dataset.plot_iq_samples(0, 1)
        # Plot the quadruplet samples
        # dataset.plot_quadruplet_samples(0)
        if saveDataFrame:
            dataset_name = "Key-Generation"
            config_name = "Sinusoid-Powder-OTA-Lab-"+"".join(str(node) for node in nodeIDs)  #"Sinusoid-Powder-OTA-Lab" 
            repo_name="CAAI-FAU"
            dataset.saveDataFrame(dataset_name, config_name, repo_name)
    
    
    # Use dataframe to plot power of the signal
    
    # Save dataframe to huggingface
    
    

