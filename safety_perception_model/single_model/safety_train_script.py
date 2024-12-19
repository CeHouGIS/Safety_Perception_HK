# python /code/LLM-crime/safety_perception_model/single_model/safety_train_script.py

from safety_train_new import *
from itertools import product
from tqdm import tqdm

variables_dict = {'lr':[0.001, 0.0001, 0.00001, 1e-6], 
                   'LLM_feature_process':['mean_dim1', 'mean_dim2', 'mean']}

combinations = list(product(*variables_dict.values()))

for combination in tqdm(combinations):
    input_dict = dict(zip(variables_dict.keys(), combination))
    input_dict['subfolder_name'] = '_'.join([f"{key}_{value}" for key, value in input_dict.items()])
    
    main(input_dict)