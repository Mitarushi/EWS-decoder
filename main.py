import pyaudio
import numpy as np
from scipy.signal import convolve, windows
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import multiprocessing as mp

from ews_decoder import FSK


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
    def __init__(self, average_window=1000):
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


class DataPlotter(pg.GraphicsLayoutWidget):
    def __init__(self, freq_points, delta_hertz, update_interval, heatmap_size):
        super().__init__()

        low_cutoff = 50
        max_amp = 1000.0
        heatmap_width = 3
        spectrogram_width = 1

        self.heatmap_size = heatmap_size
        # 二倍用意しておいて円環のときのテク
        self.heatmap_data = np.zeros((heatmap_size * 2, freq_points))

        # 対数表示のため、グラフ内のyラベルは実際の値と異なる
        self.y_pos_data = np.arange(freq_points)
        self.y_input_data = np.arange(freq_points) * delta_hertz
        self.y_real_data = np.geomspace(
            low_cutoff, freq_points * delta_hertz, num=freq_points
        )
        y_ticks = []
        pos_0_log = np.log(low_cutoff)
        pos_max_log = np.log(freq_points * delta_hertz)
        for i in range(0, 7):
            for digits in range(1, 10):
                freq = 10**i * digits
                freq_pos = round(
                    (np.log(freq) - pos_0_log) / (pos_max_log - pos_0_log) * freq_points
                )
                if 0 <= freq_pos < freq_points:
                    if digits == 1:
                        y_ticks.append((freq_pos, f"{freq}Hz"))
                    else:
                        y_ticks.append((freq_pos, ""))

        self.update_interval = update_interval
        self.update_counter = 0

        # グラフの生成
        self.setWindowTitle("FFT")
        self.resize(2400, 1100)

        self.heatmap_plot = self.addPlot(
            row=0, col=0, title="Spectrogram", colspan=heatmap_width
        )
        self.heatmap_plot.setLabel("left", "Frequency (Hz)")
        self.heatmap_plot.setLabel("bottom", "Sample")
        self.heatmap_plot.setYRange(0, freq_points)
        self.heatmap_plot.setXRange(-heatmap_size, 0)
        self.heatmap_plot.showGrid(x=True, y=True)
        self.heatmap_plot.disableAutoRange()
        self.heatmap_item = pg.ImageItem(
            self.heatmap_data[: self.heatmap_size],
            autoLevels=False,
            levels=(0, max_amp),
            rect=pg.QtCore.QRectF(-heatmap_size, 0, heatmap_size, freq_points),
        )
        self.heatmap_item.setColorMap(pg.colormap.get("CET-L16"))
        self.heatmap_plot.addItem(self.heatmap_item)
        self.heatmap_plot.setClipToView(True)
        ay = self.heatmap_plot.getAxis("left")
        ay.setTicks([y_ticks])

        self.spec_plot = self.addPlot(
            row=0, col=heatmap_width, colspan=spectrogram_width
        )
        self.spec_plot.setLabel("left", "Frequency (Hz)")
        self.spec_plot.setLabel("bottom", "Amplitude")
        self.spec_plot.setYRange(0, freq_points)
        self.spec_plot.setXRange(0, max_amp)
        self.spec_plot.showGrid(x=True, y=True)
        self.spec_plot.disableAutoRange()
        self.spec_curve = self.spec_plot.plot(
            np.zeros_like(self.y_pos_data),
            self.y_pos_data,
            pen="w",
            name="Spectrogram Curve",
        )
        self.spec_plot.setTitle("Frequency Spectrum")
        self.spec_plot.setClipToView(True)
        ay = self.spec_plot.getAxis("left")
        ay.setTicks([y_ticks])

        self.spec_plot.setYLink(self.heatmap_plot)

        # 一番下にtextを表示
        self.nextRow()
        self.info_text = pg.LabelItem(
            "Info",
            justify="center",
            size="10pt",
            color="w",
            background="k",
        )
        self.addItem(
            self.info_text, row=1, col=0, colspan=heatmap_width + spectrogram_width
        )

        for i in range(heatmap_width + spectrogram_width):
            self.ci.layout.setColumnStretchFactor(i, 1)

    def interpolate_freq_data(self, freq_data):
        return np.interp(
            self.y_real_data,
            self.y_input_data,
            freq_data,
            left=0,
            right=0,
        )

    def update_plot(self, freq_data, info_text):
        freq_data = self.interpolate_freq_data(freq_data)

        self.update_counter += 1

        heatmap_update_idx = self.update_counter % self.heatmap_size
        self.heatmap_data[heatmap_update_idx, :] = freq_data
        self.heatmap_data[heatmap_update_idx + self.heatmap_size, :] = freq_data

        if self.update_counter % self.update_interval == 0:
            self.spec_curve.setData(freq_data, self.y_pos_data)

            heatmap_data_rotate = self.heatmap_data[
                heatmap_update_idx + 1 : heatmap_update_idx + 1 + self.heatmap_size
            ]
            self.heatmap_item.setImage(heatmap_data_rotate, autoLevels=False)

            self.info_text.setText(
                f"<div style='white-space: pre; font-family: \"Courier New\", Courier, monospace;'>{info_text}</div>"
            )

            QtWidgets.QApplication.processEvents()
            return True
        return False

    def closeEvent(self, event):
        self.clear()
        QtWidgets.QApplication.instance().quit()
        event.accept()


def fft_worker_func(sampler, fft_processor, result_queue, stop_event):
    rate_monitor = LoopRateMonitor()
    while not stop_event.is_set():
        try:
            data = sampler.stream.read(sampler.chunk_size)
        except Exception as e:
            print(f"Error reading audio data: {e}")
            break
        if stop_event.is_set():
            break

        audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        audio_data = audio_data.reshape(-1, sampler.channels).mean(axis=1)

        freq_data = fft_processor.process(audio_data)

        sample_freq = rate_monitor.add_sample(time.time_ns())

        if result_queue.qsize() < 1000:
            result_queue.put((freq_data, sample_freq))


if __name__ == "__main__":
    pg.setConfigOptions(useOpenGL=True)
    app = QtWidgets.QApplication([])

    device_index = select_audio_device()
    chunk_size = 200
    sampler = AudioSampler(device_index, chunk_size=chunk_size)

    delta_hertz = 2
    delta_arg = 2 * np.pi * delta_hertz / sampler.rate
    freq_points = 2000

    plot_update_rate = 20
    update_interval = sampler.rate // chunk_size // plot_update_rate
    data_plotter = DataPlotter(
        freq_points=freq_points,
        delta_hertz=delta_hertz,
        update_interval=update_interval,
        heatmap_size=sampler.rate // chunk_size * 3,
    )
    data_plotter.show()
    data_plotter.activateWindow()

    stream = sampler.stream
    print(
        f"Sampling from device: {sampler.device_info['name']} at {sampler.rate}Hz with {sampler.channels} channels."
    )

    freq_resol_ratio = 20
    fft_processor = LowFreqFFT(
        freq_points=freq_points,
        delta_arg=delta_arg,
        chunk_size=chunk_size,
        smoothing_log=np.log(0.1) * delta_hertz / (sampler.rate * freq_resol_ratio),
    )

    result_queue = mp.Queue()
    stop_event = mp.Event()
    fft_worker = QtCore.QThread()
    fft_worker.run = lambda: fft_worker_func(
        sampler, fft_processor, result_queue, stop_event
    )
    fft_worker.start()

    fsk_decoder = FSK(
        freq_points=freq_points,
        signal_freq_list=[640, 1024],
        delta_hertz=delta_hertz,
        sample_per_bit=sampler.rate // chunk_size // 64,
        accept_freq_diff=3,
        peak_width_ratio=0.2,
        signal_noise_threshold=2.5,
        required_signal_ratio=0.1,
    )

    timer = QtCore.QTimer()

    def update_plot():
        while not result_queue.empty():
            freq_data, sample_freq = result_queue.get()

            max_amplitude = np.max(freq_data)
            peak_frequency = np.argmax(freq_data) * delta_hertz
            info_text = (
                f"Max: {max_amplitude:>8.1f}, Peak: {peak_frequency:>8.1f}Hz, "
                f"Sample Rate: {sample_freq:>8.1f}Hz, Queue Size: {result_queue.qsize():>3d}"
            )

            fsk_decoder.update(freq_data)

            if data_plotter.update_plot(freq_data, info_text):
                app.processEvents()

    timer.timeout.connect(update_plot)
    timer.start(1000 // (sampler.rate // chunk_size))
    app.exec()

    stop_event.set()
    timer.stop()

    fft_worker.quit()
    fft_worker.wait()

    # queueの処理は、難しい
    while True:
        try:
            result_queue.get(block=True, timeout=1e-3)
        except mp.queues.Empty:
            break
        except ValueError:
            break
    result_queue.close()
    result_queue.join_thread()

    sampler.close()

    exit(0)
