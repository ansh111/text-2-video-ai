# utils.py
from diffusers import StableDiffusionPipeline, StableDiffusionUpscalePipeline
import torch, os, gc, cv2
import numpy as np
from PIL import Image

# Init models
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
).to("cuda")
pipe.enable_attention_slicing()

try:
    pipe.enable_xformers_memory_efficient_attention()
except Exception as e:
    print("xformers not enabled:", e)

upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
).to("cuda")

os.makedirs("frames", exist_ok=True)

def generate_and_upscale(prompt, index, height=384, width=384):
    image = pipe(prompt, guidance_scale=8.5, num_inference_steps=40, height=height, width=width).images[0]
    torch.cuda.empty_cache(); gc.collect()
    upscaled = upscale_pipe(prompt="highly detailed, sharp", image=image).images[0]
    upscaled.save(f"frames/frame_{index:03d}.png")
    del image, upscaled
    torch.cuda.empty_cache(); gc.collect()

def blend_frames(frame1, frame2, alpha):
    arr1 = np.array(frame1).astype(np.float32)
    arr2 = np.array(frame2).astype(np.float32)
    blended = (1 - alpha) * arr1 + alpha * arr2
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def interpolate_frames(blend_steps=2):
    frame_files = sorted([f for f in os.listdir("frames") if f.endswith(".png")])
    base_frames = [Image.open(os.path.join("frames", f)) for f in frame_files]
    interpolated = []
    for i in range(len(base_frames) - 1):
        interpolated.append(base_frames[i])
        for b in range(1, blend_steps + 1):
            alpha = b / (blend_steps + 1)
            interpolated.append(blend_frames(base_frames[i], base_frames[i+1], alpha))
    interpolated.append(base_frames[-1])
    return interpolated

def save_video(frames, output_path="ai_generated.mp4", fps=15):
    width, height = frames[0].size
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video.write(frame_cv)
    video.release()
