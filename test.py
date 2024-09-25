import sounddevice as sd
import numpy as np
import pyaudio

SAMPLE_RATE = 16000
DURATION = 4
TIMEOUT_secs = 8
CHUNK_SIZE = 1024

def record_audio():
    print("Recording audio...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
    sd.wait()
    audio = np.squeeze(audio)
    print(audio)
    print(audio.shape)
    return audio

def record2():
    print("Recording audio...")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    audio = np.array([], dtype=np.float32)
    for i in range(0, int(SAMPLE_RATE / CHUNK_SIZE * DURATION)):
        data = stream.read(CHUNK_SIZE)
        audio_chunk = np.frombuffer(data, dtype=np.float32)
        audio = np.concatenate((audio, audio_chunk))

    print(audio.shape)
    print(audio)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return audio

def test_mic(x: bool) -> None:
    if x:
        a = record_audio()
    else:
        a = record2()
    rms_value = np.sqrt(np.mean(a ** 2))
    max_rms = np.sqrt(np.max(a ** 2))
    db_value = 20 * np.log10(rms_value/())

    print("RMS value:", rms_value)
    print("Max RMS: ", max_rms)
    print("dB value:", db_value)


if __name__ == "__main__":
    test_mic(True)
