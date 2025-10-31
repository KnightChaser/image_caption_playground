# caption_models.py
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

class CaptionModel:
    def __init__(self, model_id="nlpconnect/vit-gpt2-image-captioning", device=None,
                 max_length=16, num_beams=4):
        """
        Set up the image captioning model.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id).to(self.device) # type: ignore
        self.processor = ViTImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.max_length = max_length
        self.num_beams = num_beams

    def predict(self, pil_image: Image.Image) -> str:
        """
        Generate a caption for the given PIL image.
        """
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        pixel_values = self.processor(images=[pil_image], return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values,
                                         max_length=self.max_length,
                                         num_beams=self.num_beams) # type: ignore
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return preds[0].strip()
