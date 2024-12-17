import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import requests
import re

class LLMDialogueGenerator:
    def __init__(self,device="cuda:1"):
        cache_dir = "/data1/cehou_data/LLM_safety/LLM_model"
        self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        self.model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, cache_dir=cache_dir) 
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
    def dialogue(self, conversation, GSV_img=None, max_new_tokens=512):
        if GSV_img is None:
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, return_tensors="pt").to(self.device)
            
        else:
            image = Image.fromarray(GSV_img)
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # print(self.processor.decode(output[0], skip_special_tokens=True))
        return self.processor.decode(output[0], skip_special_tokens=True)

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