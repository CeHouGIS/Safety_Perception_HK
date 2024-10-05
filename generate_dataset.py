# python /code/LLM-crime/generate_dataset.py --age "_" --gender "baseline" --location "_" --event "_"
# python /code/LLM-crime/generate_dataset.py --age "30" --gender "male" --location "HongKong" --event "murder"
# python /code/LLM-crime/generate_dataset.py --age "30" --gender "female" --location "HongKong" --event "murder"

import numpy as np
import pandas as pd
import torch
import os
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from LLM_dialogue_generator import LLMDialogueGenerator
import pickle
import warnings
warnings.filterwarnings('ignore')
import argparse
import datetime
import neptune

parser = argparse.ArgumentParser(description='GSV auto download script')
parser.add_argument('--age', default=None, type=str,
                    help='age of virtual agent')
parser.add_argument('--gender', default=None, type=str,
                    help='gender of virtual agent')
parser.add_argument('--location', default=None, type=str,
                    help='location of virtual agent')
parser.add_argument('--event', default=None, type=str,
                    help='event of virtual agent for safety perception')
parser.add_argument('--sample-size', default=1000, type=int,
                    help='event of virtual agent for safety perception')
parser.add_argument('--max-new-tokens', default=512, type=int,
                    help='event of virtual agent for safety perception')

def count_characters(s):
    return len(s)

def get_img(idx,GSV_rootpath):
    
    GSV_name = GSV_metadata.iloc[idx]['panoid']
    GSV_list = [f"{GSV_rootpath}/{GSV_name[0]}/{GSV_name[1]}/{GSV_name}_{angle}.jpg" for angle in range(0, 360, 90)]
    for i,path in enumerate(GSV_list):
        if i == 0:
            GSV_img = np.array(Image.open(GSV_list[0]))
        else:
            GSV_img = np.concatenate((GSV_img, np.array(Image.open(path))), axis=1)

    # visualization
    # plt.imshow(GSV_img)
    # plt.title('GSV from original dataset')
    # plt.axis('off')
    return GSV_img

def chat_process(question_list,max_new_tokens=1024):
    for i, question in enumerate(question_list):
        # print(f"chat round {i}")
        if i == 0:
            conversation = chatbot.generate_conversation(result=None, next_question=question[0], image=None)
            answer = chatbot.dialogue(conversation,max_new_tokens=max_new_tokens)
        else:
            # print(f"conversation: {conversation}\n")
            if question[1] is not None:
                conversation = chatbot.generate_conversation(answer, question[0], image=True)
                answer = chatbot.dialogue(conversation, GSV_img=question[1],max_new_tokens=max_new_tokens)
            else:
                conversation = chatbot.generate_conversation(result=answer, next_question=question[0], image=None)
                answer = chatbot.dialogue(conversation,max_new_tokens=max_new_tokens)
                
        # Clear GPU memory
        torch.cuda.empty_cache()
        # print(answer,'\n')
        # print(f"count of characters: {count_characters(answer)}")
    return answer

def generate_dataset_unit(GSV_idx, GSV_name, GSV_rootpath, answer, profile):
    dataset_unit = {
    "GSV_idx": GSV_idx,
    "GSV_name": GSV_name,
    "GSV_path":[f"{GSV_rootpath}/{GSV_name[0]}/{GSV_name[1]}/{GSV_name}_{angle}.jpg" for angle in range(0, 360, 90)],
    "text_description": answer,
    "age":profile['age'],
    "gender": profile['gender'],
    "location": profile['location'],
    "event": profile['event']
    }
    return dataset_unit


if __name__ == '__main__':
    # Load the pre-trained LLM model
    chatbot = LLMDialogueGenerator(device='cuda:3')
    GSV_rootpath = "/data_nas/GoogleSV/images/China/HongKong"
    args = parser.parse_args()

    run = neptune.init_run(
        project="ce-hou/LLMData",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
    )  # your credentials

    # GSV dataset
    GSV_metadata_path = '/data_nas/GoogleSV/metadata/China/HongKong/pano_2024-08-17 13:11:09.125553_23512.p' # Hong Kong SVI
    GSV_metadata = pd.read_pickle(GSV_metadata_path)
    random_indices = GSV_metadata.sample(n=args.sample_size).index.tolist()
    # idx = random_indices[0]

    profile = {
        "age": args.age,
        "gender": args.gender,
        "location": args.location,
        "event": args.event
    }
    
    # answer_list = []

    print("Start generating dateset")
    dataset_list = []
    i = 0
    for idx in tqdm(random_indices):
        print(f"Processing {idx}")
        GSV_img = get_img(idx, GSV_rootpath)
        
        if args.gender == "baseline":
            question_list = [
                [f"Please design a street safety perception system rating scale and list in as much detail as possible the different information that people pay attention to in street perception by looking around the built environment (elements of the urban environment that Street View images can capture). Note that we cannot provide subjective information about residents' personal experiences, so look for key points from the objective environment, please answer this question within 500 words.", None],
                ["Based on your answers, evaluate the safety perception brought to you by the street scenes in the panoramic street view image one by one. Please answer this question within 500 words.", GSV_img],
                ]
        else:
            question_list = [
                    [f"Please design a {profile['event']}-focused street safety perception system rating scale and list in as much detail as possible the different information that people pay attention to in street perception by looking around the built environment (elements of the urban environment that Street View images can capture). Note that we cannot provide subjective information about residents' personal experiences, so look for key points from the objective environment, please answer this question within 500 words.", None],
                    ["Based on your answers, evaluate the safety perception brought to you by the street scenes in the panoramic street view image one by one. Please answer this question within 500 words.", GSV_img],
                    [f"When evaluating the safety perception of specific criminal behaviors, it is important to consider the sensitivity of different demographic groups to their environment. For {profile['gender']}, which parts of the image would you emphasize? Please answer this question within 500 words.", None],
                    [f"For individuals around the age of {profile['age']}, what factors in the image do you think would impact their sense of safety? Please answer this question within 500 words.", None],
                    [f"When discussing the safety perception in {profile['location']}, what is the characteristics of the built environment, and how these characteristics will influence people's safety perception? Please answer this question within 500 words.", None],
                    [f"In assessing safety perception regarding {profile['event']}, what do you believe are the important part of the image? Please answer this question within 500 words.", None],
                    ["Based on your answers, evaluate the safety perception brought to you by the street scenes in the panoramic street view image one by one. Please answer this question within 500 words.", None],
                    ["Taking into account all the above points, what is your overall view of the safety perception in this location? Please answer this question within 500 words.", None]
                    # ["Now, we try to focus on the females, if you were to evaluate the safety perception of women in their 30s, and rate the results from 1 to 10, with 1 being the least safe and 10 being the safest, what results would you give? Please note that if the information given in the image is insufficient to reflect the characteristics, it will be marked 'not applicable'.", GSV_img]
                ]
    
        answer = chat_process(question_list,max_new_tokens=args.max_new_tokens)
        # answer_list.append([idx, answer])
        dataset_list.append(generate_dataset_unit(idx, GSV_metadata.iloc[idx]['panoid'], GSV_rootpath, answer, profile))
        with open(f'/data_nas/cehou/LLM_safety/dataset_{args.age}_{args.gender}_{args.location}_{args.event}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}.pkl', 'wb') as f:
                pickle.dump(dataset_list, f)
        # if i % 10 == 0:
        #     with open(f'/data_nas/cehou/LLM_safety/dataset_{args.age}_{args.gender}_{args.location}_{args.event}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}.pkl', 'wb') as f:
        #         pickle.dump(dataset_list, f)
        i += 1
        
