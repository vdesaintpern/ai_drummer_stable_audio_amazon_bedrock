from pydub import AudioSegment
from huggingface_hub import hf_hub_download
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import torch
import torchaudio
from einops import rearrange

print("Loading stable audio")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
print("Loaded stable audio")
sample_rate = model_config["sample_rate"]
sample_size = int(model_config["sample_size"] / 1)

device = "cpu" 

if torch.backends.mps.is_available():
    print("MPS found")
    device = "mps"

if torch.cuda.is_available():
    print("CUDA found")
    device = "cuda"

model = model.to(device)

conditioning = [{
    "prompt": None,
    "seconds_start": 0, 
    "seconds_total": 24
}]

# This can help to avoid generating other instruments on top of drums
negative_conditioning = [{
    "prompt": "keyboard, rhodes, synth, vocals, effects, bass, congas, guitar",
    "seconds_start": 0, 
    "seconds_total": 24
}]

def generate_audio_from_prompt(prompt:str, tempo:int, bars:int) -> AudioSegment:

    conditioning[0]["prompt"] = prompt

    # Generate stereo audio
    output = generate_diffusion_cond(
        model,
        steps=50,
        cfg_scale=7,
        conditioning=conditioning,
        negative_conditioning=negative_conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")

    # Peak normalize, clip, convert to int16, and save to file
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    torchaudio.save("output.wav", output, sample_rate)

    all_audio = AudioSegment.from_wav("output.wav")

    max = 1/tempo * 60 * 4 * bars * 1000
    print(max)

    return all_audio[0:max]


