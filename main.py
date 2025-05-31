import pyaudio
import numpy as np


def select_audio_device():
    audio = pyaudio.PyAudio()
    device_count = audio.get_device_count()
    print("Available audio devices:")
    for i in range(device_count):
        device_info = audio.get_device_info_by_index(i)
        name = device_info.get('name', 'Unknown Device')
        channels = device_info.get('maxInputChannels', 0)
        rate = device_info.get('defaultSampleRate', 0)
        print(f"Index: {i}, Name: {name}, Channels: {channels}, Sample Rate: {rate}")
    
    device_index = int(input("Select an audio device index: "))
    return device_index

class AudioSampler:
    def __init__(self, device_index, chunk_size=320, sample_format=pyaudio.paInt24):
        self.audio = pyaudio.PyAudio()
        
        self.device_index = device_index
        self.device_info = self.audio.get_device_info_by_index(device_index)
        self.channels = int(self.device_info['maxInputChannels'])
        self.rate = int(self.device_info['defaultSampleRate'])
        self.chunk_size = chunk_size
        self.sample_format = sample_format

        self.setup_stream()
    
    def setup_stream(self):
        if hasattr(self, 'stream'):
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
        
        self.stream = self.audio.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
    
    def close(self):
        if hasattr(self, 'stream'):
            if self.stream.is_active():
                self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def __del__(self):
        self.close()

if __name__ == "__main__":
    device_index = select_audio_device()
    sampler = AudioSampler(device_index)

    stream = sampler.stream
    print(f"Sampling from device: {sampler.device_info['name']} at {sampler.rate}Hz with {sampler.channels} channels.")
    
    try:
        while True:
            data = stream.read(sampler.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            audio_data = audio_data.reshape(-1, sampler.channels).mean(axis=1)
            max_amplitude = np.max(np.abs(audio_data))
            print(f"Max amplitude: {max_amplitude:.2f}")
    except KeyboardInterrupt:
        print("Stopping audio sampling.")
    finally:
        sampler.close()
        print("Audio sampler closed.")
        print("Exiting program.")
            