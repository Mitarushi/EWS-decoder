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

        elapsed_time = max(1, (current_time - self.samples[0])) / 1e9

        if len(self.samples) <= 1:
            return 0
        sample_freq = (len(self.samples) - 1) / elapsed_time
        return sample_freq


class LowFreqFFT:
    def __init__(self, freq_points, delta_arg, chunk_size, smoothing_log, low_mask=50):
        self.delta_arg = delta_arg
        self.chunk_size = chunk_size
        self.smoothing_log = smoothing_log
        self.low_mask = low_mask

        n = chunk_size
        m = freq_points

        # https://chatgpt.com/share/683aee05-c9c0-8013-8bf2-14065fb9c15d
        s = smoothing_log
        w = self.delta_arg * np.arange(m)
        rho = np.exp(s)
        D = 1 - 2 * rho * np.cos(2 * w) + rho**2
        Ar = (1 - rho * np.cos(2 * w)) / D
        Ai = rho * np.sin(2 * w) / D
        Br = 1.0 / (1.0 - rho)
        self.P = Ar + Br
        self.Q = Ai
        self.S = Ar - Br
        self.Delta = (
            1 / abs(self.P * self.S + self.Q**2) * np.arange(m)
        )  # 最後のarangeはoptional

        q_log = 1j * self.delta_arg + smoothing_log
        self.qin = np.exp(q_log * np.arange(m) * n)
        self.qi2 = np.flip(np.exp(np.arange(n) ** 2 * (-q_log / 2)))
        self.qj2 = np.abs(np.exp(np.arange(m) ** 2 * (-q_log / 2)))
        self.qij2 = np.exp(np.arange(n + m - 1) ** 2 * (q_log / 2))
        self.c_ema = np.zeros(freq_points, dtype=np.complex128)

    def process(self, a):
        assert len(a) == self.chunk_size, "Input array must match chunk size."
        a = a.astype(np.complex128)

        aqi2 = a * self.qi2
        c = convolve(aqi2, self.qij2, mode="valid") * self.qj2
        self.c_ema = self.c_ema * self.qin + c

        X = np.real(self.c_ema)
        Y = np.imag(self.c_ema)
        abs_freq = (
            np.hypot(Y * self.P - X * self.Q, X * self.S + Y * self.Q) * self.Delta
        )
        return abs_freq


class PlotUpdater:
    def __init__(
        self, init_x_data, freq_points, delta_hertz, update_interval, heatmap_size
    ):
        low_cutoff = 50
        heatmap_width = 3
        spectrogram_width = 1

        self.heatmap_size = heatmap_size
        self.heatmap_data = np.zeros((heatmap_size, freq_points))

        # heatpmap (3) | spectrogram (1)
        self.fig, (self.ax_heatmap, self.ax_spec) = plt.subplots(
            nrows=1,
            ncols=2,
            gridspec_kw={
                "width_ratios": [heatmap_width, spectrogram_width],
                "wspace": 0.1,
            },
            figsize=(12, 3),
        )

        self.ax_spec.set_title("Spectrogram")
        (self.spec_plot,) = self.ax_spec.plot(
            np.zeros_like(init_x_data), init_x_data, lw=2, animated=True,
        )
        # self.ax_spec.set_ylabel("Frequency (Hz)")
        self.ax_spec.set_xlabel("Amplitude")
        self.ax_spec.set_ylim(low_cutoff, freq_points * delta_hertz)
        self.ax_spec.set_xlim(0, 1000.0)
        self.ax_spec.set_yscale("log")
        self.ax_spec.grid(True)

        self.ax_heatmap.set_title("Heatmap")
        self.heatmap_plot = self.ax_heatmap.imshow(
            self.heatmap_data.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1000.0,
            extent=(-heatmap_size, 0, 0, freq_points * delta_hertz),
            animated=True,
        )
        self.ax_heatmap.set_ylabel("Frequency (Hz)")
        self.ax_heatmap.set_xlabel("Time")
        self.ax_heatmap.set_ylim(low_cutoff, freq_points * delta_hertz)
        self.ax_heatmap.set_xlim(-heatmap_size, 0)
        self.ax_heatmap.set_yscale("log")
        self.ax_heatmap.grid(True)

        plt.ion()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        self.update_interval = update_interval
        self.update_counter = 0

    def update_plot(self, freq_data):
        self.update_counter += 1

        heatmap_update_idx = self.update_counter % self.heatmap_size
        self.heatmap_data[heatmap_update_idx, :] = freq_data

        if self.update_counter % self.update_interval == 0:
            self.fig.canvas.restore_region(self.background)
            
            self.spec_plot.set_xdata(freq_data)
            self.ax_spec.draw_artist(self.spec_plot)

            # self.heatmap_plot.set_array(np.roll(self.heatmap_data, heatmap_update_idx, axis=0).T)
            self.heatmap_plot.set_array(self.heatmap_data.T)
            self.ax_heatmap.draw_artist(self.heatmap_plot)
            
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()


if __name__ == "__main__":
    device_index = select_audio_device()
    chunk_size = 256
    sampler = AudioSampler(device_index, chunk_size=chunk_size)

    delta_hertz = 2
    delta_arg = 2 * np.pi * delta_hertz / sampler.rate
    freq_points = 2000

    plot_update_rate = 10
    update_interval = sampler.rate // chunk_size // plot_update_rate
    plot_updater = PlotUpdater(
        init_x_data=np.arange(freq_points) * delta_hertz,
        freq_points=freq_points,
        delta_hertz=delta_hertz,
        update_interval=update_interval,
        heatmap_size=sampler.rate // chunk_size * 3,
    )

    stream = sampler.stream
    print(
        f"Sampling from device: {sampler.device_info['name']} at {sampler.rate}Hz with {sampler.channels} channels."
    )

    fft_processor = LowFreqFFT(
        freq_points=freq_points,
        delta_arg=delta_arg,
        chunk_size=chunk_size,
        smoothing_log=np.log(0.1)
        * delta_hertz
        / (sampler.rate * 10),  # 周波数 / 10 ぐらいの分解能
    )

    try:
        rate_monitor = LoopRateMonitor()
        while True:
            data = stream.read(sampler.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data.reshape(-1, sampler.channels).mean(axis=1)

            freq_data = fft_processor.process(audio_data)
            print(max(freq_data))

            plot_updater.update_plot(freq_data)

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
