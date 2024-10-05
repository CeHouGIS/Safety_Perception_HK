import pandas as pd
import os
import warnings
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
from concurrent.futures import ThreadPoolExecutor, as_completed

def cal_P_and_N(select_id, category, data):
    data_test = data[(data['left_id'] == select_id) | (data['right_id'] == select_id)]

    data_test_group = data_test.groupby('category')
    # Check if the category exists in the grouped data
    if category not in data_test_group.groups:
        # raise KeyError(f"Category '{category}' not found in the data.")
        return None, None, None, None
    
    test_group = data_test_group.get_group(category)

    for i,line in test_group.iterrows():
        if line['winner'] == 'left':
            test_group.loc[i, "winner_id"] = line["left_id"]
            test_group.loc[i, "loser_id"] = line["right_id"]
        elif line['winner'] == 'right':
            test_group.loc[i, "winner_id"] = line["right_id"]
            test_group.loc[i, "loser_id"] = line["left_id"]
        else:
            test_group.loc[i, "winner_id"] = "equal"
            test_group.loc[i, "loser_id"] = "equal"

    winner_id = test_group['winner_id'].tolist()
    loser_id = test_group['loser_id'].tolist()

    p_i = len(test_group[test_group['winner_id'] == select_id]) 
    e_i = len(test_group[test_group['winner_id'] == "equal"]) 
    n_i = len(test_group) - p_i - e_i

    P_i = p_i / (p_i + e_i + n_i)
    N_i = n_i / (p_i + e_i + n_i)
    return P_i, N_i, winner_id, loser_id

def cal_Q(select_id, category, data):
    P_i, N_i, winner_id, loser_id = cal_P_and_N(select_id, category, data)
    if (P_i is None) or (N_i is None):
        return None

    sub_pi_list = []
    for i in winner_id:
        if (i != "equal") and (i != select_id):
            sub_P_i, _, _, _ = cal_P_and_N(i, category, data)
            sub_pi_list.append(sub_P_i)

    sub_ni_list = []
    for i in loser_id:
        if (i != "equal") and (i != select_id):
            _, sub_N_i, _, _ = cal_P_and_N(i, category, data)
            sub_ni_list.append(sub_N_i)

    if len(sub_pi_list) == 0:
        sub_P_i_avg = 0
    else:
        sub_P_i_avg = np.mean(sub_pi_list)
        
    if len(sub_ni_list) == 0:
        sub_n_i_avg = 0
    else:
        sub_n_i_avg = np.mean(sub_ni_list)
    Q = (1 / 3) * (P_i + sub_P_i_avg - sub_n_i_avg + 1)
    return Q

def calculate_Q_for_image(img_id):
    results = []
    for c in category:
        Q = cal_Q(img_id, c, data)
        results.append([img_id, c, Q])
    return results

if __name__ == '__main__':
    data_path = "/data_nas/cehou/LLM_safety/PlacePulse2.0/metadata/final_data.csv"
    img_path = "/data_nas/cehou/LLM_safety/PlacePulse2.0/photo_dataset/final_photo_dataset"
    data = pd.read_csv(data_path)
    img_id_ls = [i.split('.')[0] for i in os.listdir(img_path)]
    category = data['category'].value_counts().index.tolist()
    
    Q_ls = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(calculate_Q_for_image, img_id) for img_id in tqdm(img_id_ls)]
        for future in as_completed(futures):
            Q_ls.extend(future.result())
            print(len(Q_ls))
            if len(Q_ls) % 100 == 0:
                Q_df = pd.DataFrame(Q_ls, columns=['Image_ID', 'Category', 'Q_Value'])
                Q_df.to_csv("/data_nas/cehou/LLM_safety/image_perception.csv", index=False)