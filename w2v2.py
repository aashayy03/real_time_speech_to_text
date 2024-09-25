import sounddevice as sd
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import queue
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch


SAMPLE_RATE = 16000
DURATION = 4
TIMEOUT_secs = 12

def record_audio(buffer: queue.Queue, flag) -> None:
    print("Recording audio...")
    while not flag.is_set():
        audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        audio = np.squeeze(audio)
        audio = torch.FloatTensor(audio)
        buffer.put(audio)
    print("################\nRecording Stopped\n################")

def transcribe_audio(buffer: queue.Queue, model, processor) -> None:
    try:
        while True:
            audio = buffer.get(timeout = TIMEOUT_secs)
            inputs = processor(audio, sampling_rate = SAMPLE_RATE, return_tensors = 'pt', padding = True).input_values
            with torch.no_grad():
                logits = model(inputs).logits

            tokens = torch.argmax(logits, dim= -1)
            text = processor.batch_decode(tokens)[0]
            print(str(text).lower())
    except queue.Empty:
        print("################\nTranscription Stopped\n################")

def speech_to_text_wv2vec2() -> None:
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h')
    buffer = queue.Queue()
    rec_stop_flag = Event()
    with ThreadPoolExecutor() as executor:
        producer = executor.submit(record_audio, buffer, rec_stop_flag)
        consumer = executor.submit(transcribe_audio, buffer, model, processor)
        try:
            rec_stop_flag.wait()
        except KeyboardInterrupt:
            rec_stop_flag.set()
            producer.cancel()

if __name__ == "__main__":
    speech_to_text_wv2vec2()