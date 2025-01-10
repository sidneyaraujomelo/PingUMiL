import os
from glob import glob
import re
import shutil

input_path = "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_hitprediction2/Raw Data"
output_path = "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_hitprediction2/preprocessed_graphs"
input_prefix = "hitpred"
paths = glob(f"{input_path}/*")
files = [os.path.basename(x) for x in paths]
groups = [re.search(fr"({input_prefix}.*xml)", x)[0] for x in files]
groups = list(set(groups))
paths_dict = {}
for k in groups:
    paths_dict[k] = [x for x in paths if k in x]
    new_folder = os.path.join(output_path, os.path.splitext(k)[0])
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for x in paths_dict[k]:
        print(x)
        print(os.path.join(new_folder, os.path.basename(x)))
        shutil.copyfile(x, os.path.join(new_folder, os.path.basename(x)))