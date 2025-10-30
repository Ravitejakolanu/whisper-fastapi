from fastapi import FastAPI, UploadFile
from faster_whisper import WhisperModel

app = FastAPI()

model = WhisperModel("tiny", device="cpu")

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    contents = await audio.read()

    file_name = "audio.wav"
    with open(file_name, "wb") as f:
        f.write(contents)

    segments, info = model.transcribe(
        file_name, 
        language="te", 
        task="translate",
        temperature=0
    )
    text = " ".join([seg.text for seg in segments])

    return {"text": text}
