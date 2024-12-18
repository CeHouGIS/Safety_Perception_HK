import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
import torch
import pandas as pd
import numpy as np
import os

class LLaVaFeatureExtractor:
    def __init__(self,device):
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
        self.cache_dir = "/data1/cehou_data/LLM_safety/LLM_model"
        # model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, cache_dir=self.cache_dir, device_map="auto")

    def image_extractor(self, images):
        conversation_script = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                    ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation_script, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, padding=True, return_tensors="pt").to(self.model.device)
        return inputs['pixel_values']

    def text_extractor(self, text):
        conversation_script = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{text}"},
                    ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation_script, add_generation_prompt=True)
        inputs = self.processor(text=prompt, padding=True, return_tensors="pt").to(self.model.device)
        return inputs['input_ids']
    
# llm_extractor = LLaVaFeatureExtractor()
# image_features = llm_extractor.image_extractor([images])
# text_features = llm_extractor.text_extractor(text)