import pyaudio
import numpy as np
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


class SampleCounter:
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


if __name__ == "__main__":
    device_index = select_audio_device()
    chunk_size = 1500
    sampler = AudioSampler(device_index, chunk_size=chunk_size)

    stream = sampler.stream
    print(
        f"Sampling from device: {sampler.device_info['name']} at {sampler.rate}Hz with {sampler.channels} channels."
    )

    try:
        sample_counter = SampleCounter()
        while True:
            data = stream.read(sampler.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data.reshape(-1, sampler.channels).mean(axis=1)
            max_amplitude = np.max(np.abs(audio_data))

            sample_freq = sample_counter.add_sample(time.time_ns())
            print(
                f"Max Amplitude: {max_amplitude:.2f}, Sample Frequency: {sample_freq:.2f}Hz"
            )

    except KeyboardInterrupt:
        print("Stopping audio sampling.")
    finally:
        sampler.close()
        print("Audio sampler closed.")
        print("Exiting program.")
