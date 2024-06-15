
import torch
import random
from asdff.base import AdPipelineBase
from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionXLPipeline

face_prompt = "image of emma watson."
face_n_prompt = "nsfw, blurry, disfigured"
guidance_scale = 7
num_images = 1
prompt = "full body photo of emma watson in black clothes, night city street, bokeh"
n_prompt = "pencil drawing, black and white, greyscale, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
face_mask_pad = 32
mask_blur = 4
mask_dilation = 4
strength = 0.4
ddim_steps = 20

seed = random.randint(0, 3774)
generator = torch.manual_seed(seed)

model_path = r"F:\Pranav\checkpoints\stable-diffusion-xl-base-1.0\sd_xl_base_1.0.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(model_path, safety_checker=None, variant="fp16", torch_dtype=torch.float16).to("cuda")


output = pipe(
                prompt=prompt,
                negative_prompt=n_prompt,
                generator=generator,
                guidance_scale=guidance_scale,
                num_images_per_prompt=int(num_images),
                num_inference_steps=int(ddim_steps),
            )
ad_images = output.images

ad_components = pipe.components
ad_pipe = AdPipelineBase(**ad_components)

model_path = hf_hub_download("Bingsu/adetailer", "face_yolov8n.pt",local_dir = "asdff/yolo_models", local_dir_use_symlinks = False)
common = {"prompt": face_prompt,"n_prompt" : face_n_prompt, "num_inference_steps": int(ddim_steps), "target_size" : (1024,1024)}
inpaint_only = {'strength': strength}
result = ad_pipe(common=common, inpaint_only=inpaint_only, images=ad_images, mask_dilation=mask_dilation, mask_blur=mask_blur, mask_padding=face_mask_pad, model_path=model_path)

ad_images[0].save("image.png")
if result.images:
    result.images[0].save(f'image_fix.png')

            