# generate baseline data
# python /code/LLM-crime/generate_dataset.py --visible-device "3" --age "baseline" --gender "baseline" --location "baseline" --event "baseline" --img-type "Stockholm" --start-from 0 --data-num 5000 --batch-size 10
# python /code/LLM-crime/generate_dataset.py --visible-device "2" --age "30" --gender "male" --location "Stockholm, Sweden" --event "theft or robbery" --img-type "Stockholm" --start-from 0 --data-num 5000 --batch-size 4
# python /code/LLM-crime/generate_dataset.py --visible-device "1" --age "60" --gender "male" --location "Stockholm, Sweden" --event "theft or robbery" --img-type "Stockholm" --start-from 0 --data-num 5000 --batch-size 4
# python /code/LLM-crime/generate_dataset.py --visible-device "0" --age "60" --gender "female" --location "Stockholm, Sweden" --event "theft or robbery" --img-type "Stockholm" --start-from 3840 --data-num 5000 --batch-size 4


# python /code/LLM-crime/generate_dataset.py --visible-device "1" --age "baseline" --gender "baseline" --location "baseline" --event "baseline" --img-type "GSV" --start-from 0 --data-num 4989 --batch-size 6
# python /code/LLM-crime/generate_dataset.py --visible-device "2" --age "30" --gender "male" --location "HongKong" --event "murder" --img-type "GSV" --start-from 1900 --data-num 4989 --batch-size 4

# python /code/LLM-crime/generate_dataset.py --visible-device "2,3" --age "60" --gender "female" --location "HongKong" --event "murder" --img-type "GSV" --start-from 1900 --data-num 4989 --batch-size 4
# python /code/LLM-crime/generate_dataset.py --visible-device "2,3" --age "60" --gender "male" --location "HongKong" --event "traffic accident" --img-type "GSV" --start-from 0 --data-num 4989 --batch-size 4
# python /code/LLM-crime/generate_dataset.py --visible-device "2,3" --age "60" --gender "female" --location "HongKong" --event "traffic accident" --img-type "GSV" --start-from 100 --data-num 4989 --batch-size 4

# generate specific data
# python /code/LLM-crime/generate_dataset.py --device-id "cuda:2" --age "30" --gender "female" --location "HongKong" --event "murder" --specific-img True --img-type "PlacePulse" --start-from 351
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
parser.add_argument('--visible-device', default="0,1,2,3", type=str,
                    help='event of virtual agent for safety perception')
parser.add_argument('--age', default='30', type=str,
                    help='age of virtual agent')
parser.add_argument('--gender', default='baseline', type=str,
                    help='gender of virtual agent')
parser.add_argument('--location', default='Hong Kong', type=str,
                    help='location of virtual agent')
parser.add_argument('--event', default='traffic accident', type=str,
                    help='event of virtual agent for safety perception')
parser.add_argument('--specific-img', default=False, type=bool,
                    help='event of virtual agent for safety perception')
parser.add_argument('--img-type', default='GSV', type=str,
                    help='GSV or PlacePulse')
parser.add_argument('--reference-dataset', default="/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_501.pkl", type=str,
                    help='event of virtual agent for safety perception')
parser.add_argument('--metadata-path', default="/data2/cehou/LLM_safety/Stockholm/safety_score_dataset_creating.csv", type=str,
                    help='event of virtual agent for safety perception')
parser.add_argument('--start-from', default=0, type=int,
                    help='event of virtual agent for safety perception')
parser.add_argument('--sample-size', default=200, type=int,
                    help='event of virtual agent for safety perception')
parser.add_argument('--max-new-tokens', default=512, type=int,
                    help='event of virtual agent for safety perception')
parser.add_argument('--data-num', default=5000, type=int,
                    help='event of virtual agent for safety perception')
parser.add_argument('--batch-size', default=4, type=int,
                    help='event of virtual agent for safety perception')

def count_characters(s):
    return len(s)

# def get_img(GSV_metadata, idx, GSV_rootpath, img_type):
#     if img_type == 'GSV':
#         GSV_name = GSV_metadata.iloc[idx]['panoid']
#         # GSV_list = [f"{GSV_rootpath}/{GSV_name[0]}/{GSV_name[1]}/{GSV_name}_{angle}.jpg" for angle in range(0, 360, 90)]
#         GSV_list = [f"{GSV_rootpath}/{GSV_name}_{angle}.jpg" for angle in range(0, 360, 90)]
#         for i,path in enumerate(GSV_list):
#             if i == 0:
#                 GSV_img = np.array(Image.open(GSV_list[0]))
#             else:
#                 GSV_img = np.concatenate((GSV_img, np.array(Image.open(path))), axis=1)

#         # visualization
#         # plt.imshow(GSV_img)
#         # plt.title('GSV from original dataset')
#         # plt.axis('off')
#         return GSV_img
    
def get_img(GSV_metadata, GSV_rootpath, idx, img_size, img_type='GSV'):
    GSV_name = GSV_metadata.iloc[idx]['panoid']
    if img_type == 'PlacePulse':
        GSV_img = np.array(Image.open(f"{GSV_rootpath}/{GSV_name}.jpg"))
    if img_type == 'Stockholm':
        GSV_img = np.array(Image.open(f"{GSV_rootpath}/{GSV_name}.jpg"))
    if img_type == 'GSV':
        GSV_list = [f"{GSV_rootpath}/{GSV_name[0]}/{GSV_name[1]}/{GSV_name}_{angle}.jpg" for angle in range(0, 360, 90)]
        for i,path in enumerate(GSV_list):
            if i == 0:
                GSV_img = np.array(Image.open(GSV_list[0]))
            else:
                GSV_img = np.concatenate((GSV_img, np.array(Image.open(path))), axis=1)
        
        GSV_img = np.array(Image.fromarray(GSV_img).resize((img_size[0], img_size[1])))
    # visualization
    # plt.imshow(GSV_img)
    # plt.title('GSV from original dataset')
    # plt.axis('off')
    return GSV_img

def chat_process_batch(question_list, sub_range, GSV_imgs, max_new_tokens=1024):
    for j in range(len(question_list[0])):
        print(f"chat round {j}")
        if j == 0:
            conversation_batch = [chatbot.generate_conversation(result=None, next_question=question_list[i][0], image=True) for i in range(len(sub_range))]
            answer_batch = chatbot.dialogue_batch(conversation_batch, GSV_imgs, len(sub_range), max_new_tokens=max_new_tokens)
        else:
            conversation_batch = [chatbot.generate_conversation(answer_batch[i], next_question=question_list[i][j], image=True) for i in range(len(sub_range))] 
            answer_batch = chatbot.dialogue_batch(conversation_batch, GSV_imgs, len(sub_range), max_new_tokens=max_new_tokens)

        # Clear GPU memory
        torch.cuda.empty_cache()
        # print(answer,'\n')
        # print(f"count of characters: {count_characters(answer)}")
    return answer_batch

def generate_dataset_block(GSV_idx, GSV_name, GSV_rootpath, answers, profile, img_type):

    data_block = []
    for i in range(len(answers)):
        if profile['gender'] == 'baseline':
            answer_baseline = answers[i].split('<\\s>')[1].split(' [/INST] ')[1] # gender
            if img_type == 'GSV':
                dataset_unit = {
                "GSV_idx": GSV_idx[i],
                "panoid":GSV_name[i],
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event'],
                "text_description_all": answers,
                "text_description_baseline": answer_baseline,
                }
            elif img_type == 'PlacePulse':
                dataset_unit = {
                "GSV_idx": GSV_idx,
                "GSV_name": GSV_name,
                "panoid":f"{GSV_rootpath}/{GSV_name}.jpg",
                "text_description": answers,
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event']
                }
            elif img_type == 'Stockholm':
                dataset_unit = {
                "GSV_idx": GSV_idx,
                "GSV_name": GSV_name,
                "panoid":f"{GSV_rootpath}/{GSV_name}.jpg",
                "text_description": answers,
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event']
                }
        else:
            answer_gender = answers[i].split('<\\s>')[1].split(' [/INST] ')[1] # gender
            answer_age = answers[i].split('<\\s>')[2].split(' [/INST] ')[1] # age
            answer_location = answers[i].split('<\\s>')[3].split(' [/INST] ')[1] # location

            if img_type == 'GSV':
                dataset_unit = {
                "GSV_idx": GSV_idx[i],
                "panoid":GSV_name[i],
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event'],
                "text_description_all": answers,
                "text_description_age": answer_age,
                "text_description_gender": answer_gender,
                "text_description_location": answer_location,
                }
            elif img_type == 'PlacePulse':
                dataset_unit = {
                "GSV_idx": GSV_idx,
                "GSV_name": GSV_name,
                "panoid":f"{GSV_rootpath}/{GSV_name}.jpg",
                "text_description": answers,
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event']
                }
            elif img_type == 'Stockholm':
                dataset_unit = {
                "GSV_idx": GSV_idx,
                "GSV_name": GSV_name,
                "panoid":f"{GSV_rootpath}/{GSV_name}.jpg",
                "text_description": answers,
                "age":profile['age'],
                "gender": profile['gender'],
                "location": profile['location'],
                "event": profile['event']
                }

        data_block.append(dataset_unit)

    return data_block

def get_qa(text_description):
    answer = [i.split("<\\s> [INST]")[0] for i in text_description.split(" [/INST] ")][1:]
    question = [text_description.split(" [/INST] ")[0][7:]] + [i.split("<\\s> [INST]")[1] for i in text_description.split(" [/INST] ")[1:-1]]
    return answer, question

def text_processing(dataset_path, data_type):
    '''
    data_type: 'baseline' or 'others'
    example: text_processing("/data1/cehou_data/LLM_safety/img_text_data/dataset_baseline_baseline_baseline_baseline_1401.pkl", 'baseline')
    '''
    dataset = pd.read_pickle(dataset_path)
    text_description = [dataset[i]['text_description'] for i in range(len(dataset))]

    answer_list = [get_qa(text_description[i])[0] for i in range(len(text_description))]
    question_list = [get_qa(text_description[i])[1] for i in range(len(text_description))]
    if data_type == 'baseline':
        text_description_new = [answer_list[i][1] for i in range(len(text_description))]
        for i in range(len(dataset)):
            dataset[i]['text_description_short'] = text_description_new[i]
    else:
        print('not implemented')

    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    return None

if __name__ == '__main__':
    # Load the pre-trained LLM model

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device

    chatbot = LLMDialogueGenerator()

    # run = neptune.init_run(
    #     project="ce-hou/LLMData",
    #     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJmYzFmZTZkYy1iZmY3LTQ1NzUtYTRlNi1iYTgzNjRmNGQyOGUifQ==",
    # )  # your credentials

    # GSV dataset
    if args.img_type == 'GSV':
        GSV_rootpath = "/data2/cehou/LLM_safety/GSV/HK_imgs"
        GSV_metadata_path = args.metadata_path
        # GSV_metadata_path = '/data2/cehou/LLM_safety/GSV/GSV_metadata_sampled_5000.csv' # Hong Kong SVI
        # GSV_metadata_path = '/data2/cehou/LLM_safety/img_text_data/data_need_generated/dataset_30_male_HongKong_traffic accident_GSV_notprocess_271.csv' # Hong Kong SVI
        GSV_metadata = pd.read_csv(GSV_metadata_path)
    elif args.img_type == 'PlacePulse':
        GSV_rootpath = "/data2/cehou/LLM_safety/PlacePulse2.0/photo_dataset/final_photo_dataset"
        GSV_metadata_path = args.metadata_path
        # GSV_metadata_path = '/data2/cehou/LLM_safety/PlacePulse2.0/train_data_need_label.csv' # Place Pulse SVI
        # GSV_metadata_path = "/data2/cehou/LLM_safety/PlacePulse2.0/train_data_5264_notprocess.csv"
        GSV_metadata = pd.read_csv(GSV_metadata_path) 
    elif args.img_type == 'Stockholm':
        GSV_rootpath = "/data2/cehou/LLM_safety/Stockholm/GSV_5000_2"
        GSV_metadata_path = args.metadata_path
        # GSV_metadata_path = '/data2/cehou/LLM_safety/PlacePulse2.0/train_data_need_label.csv' # Place Pulse SVI
        # GSV_metadata_path = "/data2/cehou/LLM_safety/PlacePulse2.0/train_data_5264_notprocess.csv"
        GSV_metadata = pd.read_csv(GSV_metadata_path) 
        
    if args.specific_img == False:  
        # random_indices = GSV_metadata.sample(n=args.sample_size).index.tolist()
        random_indices = range(args.data_num)
    else:
        print("Using specific image index")
        ls = pd.read_pickle(args.reference_dataset)
        random_indices = [i["GSV_idx"] for i in ls]

    profile = {
        "age": args.age,
        "gender": args.gender,
        "location": args.location,
        "event": args.event
    }
    print(profile)
    # answer_list = []

    print("Start generating dateset")
    dataset_list = []
    img_size = [400,200]
    select_range = np.arange(args.start_from, args.data_num, args.batch_size)
    if select_range[-1] != args.data_num:
        select_range = np.append(select_range, args.data_num)

    for idx in tqdm(range(len(select_range)-1)):
        sub_range = np.arange(select_range[idx], select_range[idx+1])
        print(f"Processing {sub_range}")
        if args.img_type == 'PlacePulse':
            GSV_imgs = [Image.fromarray(get_img(GSV_metadata, GSV_rootpath, i, img_size, "PlacePulse")) for i in sub_range]
        if args.img_type == 'Stockholm':
            GSV_imgs = [Image.fromarray(get_img(GSV_metadata, GSV_rootpath, i, img_size, "Stockholm")) for i in sub_range]
        elif args.img_type == 'GSV':
            GSV_imgs = [get_img(GSV_metadata, GSV_rootpath, i, img_size, "GSV") for i in sub_range]

        if args.gender == "baseline":
            print("baseline")
            question_list =[[
                [f"Please design a street safety perception system rating scale and list in as much detail as possible the different information that people pay attention to in street perception by looking around the built environment (elements of the urban environment that Street View images can capture). Note that we cannot provide subjective information about residents' personal experiences, so look for key points from the objective environment, please answer this question within 300 words.", GSV_imgs[i]],
                ["Based on your answers, evaluate the safety perception brought to you by the street scenes in the panoramic street view image one by one. Please answer this question within 300 words.", GSV_imgs[i]],
                ] for i in range(len(sub_range))]
        else:
        #     question_list = [
        #         [f"Please design a {profile['event']}-focused street safety perception system rating scale and list briefly and include different information that people pay attention to in street perception by looking around the built environment (elements of the urban environment that Street View images can capture). Please pay special attention to the fact that people of different ages and genders may have different perceptions and reactions. Note that we cannot provide subjective information about residents' personal experiences, so look for key points from the objective environment, please answer this question within 300 words.", GSV_img],
        #         [f"When evaluating the safety perception of specific criminal behaviors, it is important to consider the sensitivity of different demographic groups to their environment. For {profile['gender']}, which parts of the image would you emphasize? Please answer this question within 300 words.", GSV_img],
        #         [f"For individuals around the age of {profile['age']}, what factors in the image do you think would impact their sense of safety? Please answer this question within 300 words.", GSV_img],
        #         [f"When discussing the safety perception in {profile['location']}, what is the characteristics of the built environment, and how these characteristics will influence people's safety perception? Please answer this question within 300 words.", GSV_img]
        # ]

            question_list = [[
                [f"Please design a car accident-focused street safety perception system list briefly and include different information that people pay attention to in street perception by looking around the built environment (elements of the urban environment that Street View images can capture). Please pay special attention to the fact that people of different ages and genders may have different perceptions and reactions. Note that we cannot provide subjective information about residents' personal experiences, so look for key points from the objective environment, please answer this question within 300 words."],
                [f"When evaluating the safety perception of specific criminal behaviors, it is important to consider the sensitivity of different demographic groups to their environment. For {profile['gender']} as a pedestrian, which parts of the image would you emphasize? Please answer this question within 300 words.", GSV_imgs[i]],
                [f"For people in the age of {profile['age']}, what factors in the image do you think would impact their sense of safety? Please answer this question within 300 words."],
                [f"When discussing the safety perception in {profile['location']}, what is the characteristics of the built environment, and how these characteristics will influence people's safety perception? Please answer this question within 300 words."]
            ] for i in range(len(sub_range))]

        answers = chat_process_batch(question_list, sub_range, GSV_imgs, max_new_tokens=args.max_new_tokens)
        print(answers, len(answers))
        # answer_list.append([idx, answer])
        dataset_list.append(pd.DataFrame(generate_dataset_block(sub_range, list(GSV_metadata.iloc[sub_range]['panoid']), GSV_rootpath, answers, profile, 'GSV')))
        if idx % 5 == 0:
            tem = pd.DataFrame()
            for i in range(len(dataset_list)):
                tem = pd.concat([tem, dataset_list[i]], axis=0)
            tem = tem.reset_index(drop=True)
            with open(f'/data2/cehou/LLM_safety/img_text_data/{args.img_type}_dataset_{args.age}_{args.gender}_{args.location}_{args.event}_{args.img_type}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.start_from}_{idx}.pkl', 'wb') as f:
                pickle.dump(tem, f)
        # if i % 10 == 0:
        #     with open(f'/data_nas/cehou/LLM_safety/dataset_{args.age}_{args.gender}_{args.location}_{args.event}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}.pkl', 'wb') as f:
        #         pickle.dump(dataset_list, f)
    tem = pd.DataFrame()
    for i in range(len(dataset_list)):
        tem = pd.concat([tem, dataset_list[i]], axis=0)
    tem = tem.reset_index(drop=True)
    with open(f'/data2/cehou/LLM_safety/img_text_data/{args.img_type}_dataset_{args.age}_{args.gender}_{args.location}_{args.event}_{args.img_type}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_{args.start_from}_{idx}.pkl', 'wb') as f:
        pickle.dump(tem, f)
        
