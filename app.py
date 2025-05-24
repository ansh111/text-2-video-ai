# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from utils import generate_and_upscale, interpolate_frames, save_video

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_video(req: PromptRequest):
    prompt = req.prompt
    for i in range(4):
        generate_and_upscale(prompt, i)
    frames = interpolate_frames(2)
    save_video(frames)
    return {"video_url": "https://hf.space/embed/YOUR_USERNAME/YOUR_SPACE_NAME/file/ai_generated.mp4"}
