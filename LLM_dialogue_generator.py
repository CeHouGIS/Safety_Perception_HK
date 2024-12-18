import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoModelForImageTextToText, BitsAndBytesConfig
from PIL import Image
import requests
import re

class LLMDialogueGenerator:
    def __init__(self):
        self.cache_dir = "/data1/cehou_data/LLM_safety/LLM_model"

        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

        # specify how to quantize the model
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=self.quantization_config, cache_dir=self.cache_dir, low_cpu_mem_usage=True, device_map="auto")
        # self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_dir) 
        # self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # print(f"Using device: {self.device}")
        # self.model.to(self.device)
        

    def dialogue_batch(self, conversation_batch, GSV_imgs, batch_size, max_new_tokens=512):
        prompts_batch = [self.processor.apply_chat_template(conversation_batch[i], add_generation_prompt=True) for i in range(batch_size)]
        inputs = self.processor(images=GSV_imgs, text=prompts_batch, padding=True, return_tensors="pt").to(self.model.device, torch.float16)
        generate_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        output_batch = self.processor.batch_decode(generate_ids, skip_special_tokens=True)
        return output_batch

    def generate_conversation(self, result, next_question, image=False):
        if result == None:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                            {
                            "type": "text",
                            "text": next_question
                        }
                    ],
                },
            ]
        else:
            result = result.split('<\\s>')
            # print("=========================")    
            # for i in result:
            #     print(len(i))
            # print("=========================")    
            conversation = []
            for i in result:
                if len(i) < 50: # remove the error generation
                    continue
                else:
                    tem = [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": i.split(' [/INST] ')[0][7:]},
                                ],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": i.split(' [/INST] ')[1]},
                                ],
                        }]
                    conversation.extend(tem)
                    
            if image != False:
                conversation.extend(
                [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text":next_question
                        },
                    ],
                }])            
            else:
                conversation.extend(
                    [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": next_question
                            },
                        ],
                    }]
                )
        return conversation