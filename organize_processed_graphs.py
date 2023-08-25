import os
from glob import glob
import re

input_path = "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_winprediction/Raw Data"
output_path = "/raid/home/smelo/PingUMiL-pytorch/dataset/SmokeSquadron/ss_winprediction/preprocessed_graphs"
paths = glob(f"{input_path}/*")
files = [os.path.basename(x) for x in paths]
groups = [re.search(r"(winpred.*xml)", x)[0] for x in files]
groups = list(set(groups))
paths_dict = {}
for k in groups:
    paths_dict[k] = [x for x in paths if k in x]
    new_folder = os.path.join(output_path, os.path.splitext(k)[0])
    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for x in paths_dict[k]:
        os.replace(x, os.path.join(new_folder, os.path.basename(x)))