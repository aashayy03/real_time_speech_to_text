import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import queue
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
DURATION = 4
TIMEOUT_secs = 12

def record_audio(buffer: queue.Queue, flag) -> None:
    print("Recording audio...")
    while not flag.is_set():
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        audio = np.squeeze(audio)
        buffer.put(audio)
    print("################\nRecording Stopped\n################")

def transcribe_audio(buffer: queue.Queue, model) -> None:
    try:
        while True:
            audio = buffer.get(timeout = TIMEOUT_secs)
            segments, info = model.transcribe(audio)
            for segment in segments:
                print(f"{segment.text}")
    except queue.Empty:
        print("################\nTranscription Stopped\n################")

def speech_to_text() -> None:
    model = WhisperModel('medium.en', device='cuda')
    buffer = queue.Queue()
    rec_stop_flag = Event()
    with ThreadPoolExecutor() as executor:
        producer = executor.submit(record_audio, buffer, rec_stop_flag)
        consumer = executor.submit(transcribe_audio, buffer, model)
        try:
            rec_stop_flag.wait()
        except KeyboardInterrupt:
            rec_stop_flag.set()
            producer.cancel()

if __name__ == "__main__":
    speech_to_text()