from fastapi import FastAPI, UploadFile
import whisper

app = FastAPI()
model = whisper.load_model("base")  # works on CPU but slow for 10+ sec audio

@app.post("/transcribe")
async def transcribe(audio: UploadFile):
    contents = await audio.read()
    file_name = "audio.wav"
    with open(file_name, "wb") as f:
        f.write(contents)

    result = model.transcribe(
        file_name,
        language="te",
        task="translate"   # Telugu speech -> English text ✅
    )

    return {"text": result["text"]}

