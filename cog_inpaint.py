from io import BytesIO
import os
import cog
from omegaconf import OmegaConf
from PIL import Image
import numpy as np
from random import randint
import torch
from torch.nn.functional import interpolate
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image_file, mask_region, device):
    pil_image = Image.open(image_file).convert("RGB")
    image = np.array(pil_image)
    bw_image = np.array(pil_image.convert('L'))
    image = image.astype(np.float32)/255.0
    bw_image = bw_image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    mask = make_mask(bw_image, mask_region)
    image = torch.from_numpy(image)
    masked_image = (1 - mask) * image
    image = image * 2.0 - 1.0
    masked_image = masked_image * 2.0 - 1.0
    mask = mask * 2.0 - 1.0
    return  {"image": image.to(device), "mask": mask.to(device), "masked_image": masked_image.to(device)}


def make_mask(image, mask_region):
    size = image.shape
    mask = np.zeros_like(image[None, None])
    cutoff = size[0] // 4
    x_rand = randint(0, size[0] - cutoff)
    y_rand = randint(0, size[1] - cutoff)
    mask_x = (x_rand, x_rand + cutoff)
    mask_y = (y_rand, y_rand + cutoff)
    mask[:, :, mask_x[0]:mask_x[1], mask_y[0]:mask_y[1]]
    """
    if mask_region == 'top':
        cutoff = size[0] // 2
        mask[:, :, :cutoff, :] = 1
    elif mask_region == 'bottom':
        cutoff = size[0] // 2
        mask[:, :, cutoff:, :] = 1
    elif mask_region == 'left':
        cutoff = size[1] // 2
        mask[:, :, :, :cutoff] = 1
    elif mask_region == 'right':
        cutoff = size[1] // 2
        mask[:, :, :, cutoff:] = 1
    else:
        raise ValueError(f"Don't know how to mask region '{mask_region}'")
    """
    return torch.from_numpy(mask)


def tensor_to_image(image_tensor):
    image_array = image_tensor.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255
    return Image.fromarray(image_array.astype(np.uint8))


class InpaintPredictor(cog.BasePredictor):
    def setup(self):
        self.config = OmegaConf.load("models/ldm/inpainting_big/config.yaml")
        model = instantiate_from_config(self.config.model)
        model.load_state_dict(
            torch.load("models/ldm/inpainting_big/last.ckpt")["state_dict"],
            strict=False
        )
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device)
        self.sampler = DDIMSampler(model)
        
    def predict(
        self,
        image: cog.Path = cog.Input(description='The image that will be inpainted.'),
        steps: int = cog.Input(description='The number of steps to use for the diffusion sampling', default=50, lt=200, gt=0),
        mask_region: str = cog.Input(description='Which half of the image to mask.', default='top', choices=['top', 'bottom', 'left', 'right'])
    ) -> cog.Path:
        with torch.no_grad(), self.model.ema_scope():
            batch = make_batch(image, mask_region, device=self.device)
            width = batch['image'].shape[-2]
            height = batch['image'].shape[-1]
            # encode masked image and concat downsampled mask
            masked_image = batch["masked_image"]
            c = self.model.cond_stage_model.encode(batch["masked_image"])
            cc = interpolate(batch["mask"], size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)
            shape = (c.shape[1] - 1, ) + c.shape[2:]
            samples_ddim, _ = self.sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=c.shape[0],
                shape=shape,
                verbose=False
            )
            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            image = torch.clamp(
                (batch["image"] + 1.0) / 2.0,
                min=0.0,
                max=1.0
            )
            mask = torch.clamp(
                (batch["mask"] + 1.0) / 2.0,
                min=0.0,
                max=1.0
            )
            predicted_image = torch.clamp(
                (x_samples_ddim + 1.0) / 2.0,
                min=0.0,
                max=1.0
            )
            masked_image = (1 - mask) * image
            inpainted = masked_image + mask * predicted_image
            output_image = Image.new(mode='RGB', size = (width * 3, height))
            output_image.paste(tensor_to_image(image), (0, 0))
            output_image.paste(tensor_to_image(masked_image), (width, 0))
            output_image.paste(tensor_to_image(inpainted), (width * 2, 0))
            output_path = './output_image.png'
            output_image.save(output_path)
            return cog.Path(output_path)

