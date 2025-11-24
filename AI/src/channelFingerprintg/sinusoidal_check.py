import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as ss


def _gen_single_channel_spectrogram(sig, win_len=256, overlap=128):
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
    f, t, spec = ss.stft(sig,
                            fs=1,
                            window='hann', 
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
    
    
# Create a sample signal
fs = 1000000  # Sampling frequency
numSamples = 8192
t = numSamples
signalFreq = 1000
signal = np.sin(2 * np.pi * signalFreq * t) #+ np.sin(2 * np.pi * 20 * t)
print("signal shape: ", signal.shape)
noise = np.random.normal(0, 0.1, numSamples)
signal = signal + noise

# Plot time domain signal
plt.figure(figsize=(10, 4))
plt.plot(t, signal)
plt.title('Time Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Compute FFT
fft_signal = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), d=1/fs)

# Plot without fftshift
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(frequencies, np.abs(fft_signal))
plt.title('FFT without fftshift')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Apply fftshift and plot
shifted_fft_signal = np.fft.fftshift(fft_signal)
shifted_frequencies = np.fft.fftshift(frequencies)

plt.subplot(1, 2, 2)
plt.plot(shifted_frequencies, np.abs(shifted_fft_signal))
plt.title('FFT with fftshift')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.tight_layout()
plt.show()

# plot the spectrogram
spectrogram = _gen_single_channel_spectrogram(signal)
plt.figure(figsize=(10, 4))
plt.imshow(spectrogram, aspect='auto', origin='lower')
plt.title('Spectrogram')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Time (s)')
plt.tight_layout()
plt.show()
exit()