import pyaudio
import numpy as np
from scipy.signal import convolve, windows
import matplotlib.pyplot as plt
import time


def select_audio_device():
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    print("Available audio devices:")
    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        name = device_info.get("name", "Unknown Device")
        channels = device_info.get("maxInputChannels", 0)
        rate = device_info.get("defaultSampleRate", 0)
        print(f"Index: {i}, Name: {name}, Channels: {channels}, Sample Rate: {rate}")

    device_index = int(input("Select an audio device index: "))
    return device_index


class AudioSampler:
    def __init__(self, device_index, chunk_size, sample_format=pyaudio.paInt16):
        self.audio = pyaudio.PyAudio()

        self.device_index = device_index
        self.device_info = self.audio.get_device_info_by_index(device_index)
        self.channels = int(self.device_info["maxInputChannels"])
        self.rate = int(self.device_info["defaultSampleRate"])
        self.chunk_size = chunk_size
        self.sample_format = sample_format

        self.setup_stream()

    def setup_stream(self):
        if hasattr(self, "stream"):
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()

        self.stream = self.audio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

    def close(self):
        if hasattr(self, "stream"):
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def __del__(self):
        self.close()


class LoopRateMonitor:
    def __init__(self, average_window=50):
        self.average_window = average_window
        self.samples = []
        self.last_sample_time = time.time_ns()

    def add_sample(self, sample):
        current_time = time.time_ns()
        self.samples.append(sample)
        if len(self.samples) > self.average_window:
            self.samples.pop(0)

        elapsed_time = (current_time - self.samples[0]) / 1e9

        if len(self.samples) <= 1:
            return 0
        sample_freq = (len(self.samples) - 1) / elapsed_time
        return sample_freq


class LowFreqFFT:
    def __init__(self, freq_points, delta_arg, chunk_size, smoothing_log, low_mask=10):
        self.delta_arg = delta_arg
        self.chunk_size = chunk_size
        self.smoothing_log = smoothing_log
        self.low_mask = low_mask
        
        self.n = chunk_size
        self.m = freq_points
    
        self.q_log = 1j * self.delta_arg + smoothing_log / self.n
        self.qin = np.exp(
            np.arange(self.m) * (1j * self.delta_arg * self.n) +
            np.maximum(self.low_mask, np.arange(self.m)) * self.smoothing_log)
        self.qi2 = np.flip(np.exp(np.arange(self.n) ** 2 * (-self.q_log / 2)))
        self.qij2 = np.exp(np.arange(self.n + self.m - 1) ** 2 * (self.q_log / 2))
        self.c_ema = np.zeros(freq_points, dtype=np.complex128)
    
    def process(self, a):
        assert len(a) == self.chunk_size, "Input array must match chunk size."
        a = a.astype(np.complex64)
        
        aqi2 = a * self.qi2
        c = convolve(aqi2, self.qij2, mode="valid") / self.n
        self.c_ema = self.c_ema * self.qin + c
        return np.abs(self.c_ema)


# def low_freq_fft(a, freq_points, delta_arg):
#     a = a.astype(np.complex64)
#     n = len(a)
#     m = freq_points
    
#     qi2 = np.exp(np.arange(n) ** 2 * (-1j * delta_arg / 2))
#     aqi2 = np.flip(a * qi2)
#     qij = np.exp(np.arange(n + m - 1) ** 2 * (1j * delta_arg / 2))
#     c = convolve(aqi2, qij, mode="valid")
#     return np.abs(c) / n


if __name__ == "__main__":
    device_index = select_audio_device()
    chunk_size = 1500
    sampler = AudioSampler(device_index, chunk_size=chunk_size)
    
    delta_hertz = 0.25
    delta_arg = 2 * np.pi * delta_hertz / sampler.rate
    freq_points = 10000
    
    # initialize plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_data = np.arange(freq_points) * delta_hertz
    data_plot, = plt.plot(x_data, np.zeros_like(x_data), lw=2)
    
    plt.xlim(0, freq_points * delta_hertz)
    plt.ylim(0, 1000.0)
    
    plt.show(block=False)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    stream = sampler.stream
    print(
        f"Sampling from device: {sampler.device_info['name']} at {sampler.rate}Hz with {sampler.channels} channels."
    )
    
    fft_processor = LowFreqFFT(
        freq_points=freq_points,
        delta_arg=delta_arg,
        chunk_size=chunk_size,
        smoothing_log=np.log(0.7) / 640,
    )

    try:
        rate_monitor = LoopRateMonitor()
        while True:
            data = stream.read(sampler.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data.reshape(-1, sampler.channels).mean(axis=1)

            freq_data = fft_processor.process(audio_data)
    
            data_plot.set_ydata(freq_data)
            fig.canvas.restore_region(background)
            ax.draw_artist(data_plot)
            fig.canvas.blit(fig.bbox)
            fig.canvas.flush_events()

            max_amplitude = np.max(np.abs(audio_data))
            peak_frequency = np.argmax(freq_data) * delta_hertz
            sample_freq = rate_monitor.add_sample(time.time_ns())
            print(
                f"Max Amp: {max_amplitude:.2f}, Peak Freq: {peak_frequency:.2f}Hz, Sample Freq: {sample_freq:.2f}Hz"
            )

    except KeyboardInterrupt:
        print("Stopping audio sampling.")
    finally:
        sampler.close()
        print("Audio sampler closed.")
        print("Exiting program.")
