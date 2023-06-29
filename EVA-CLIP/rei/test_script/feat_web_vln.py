import torch
from eva_clip import create_model_and_transforms, get_tokenizer
from PIL import Image
from tqdm import tqdm
import os
import pickle as pkl

model_name = "EVA02-CLIP-bigE-14-plus"  
model_name = "EVA02-CLIP-bigE-14"
model_name = "EVA02-CLIP-B-16" 
pretrained = "eva_clip" # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"


feats={}
device = "cuda" #if torch.cuda.is_available() #else "cpu"
model, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
model = model.to(device)
dat_dir = f"/workspace/code/webvln/imgs"
for root, dirs, files in os.walk(dat_dir):
    for file in tqdm(files):
        image_path = f"{dat_dir}/{file}"
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = model.encode_image(image)
            feats[file.split(".")[0]] = image_features.cpu().numpy()

f = open("/workspace/code/webvln/img_feats.pkl","wb")
pkl.dump(feats,f)
f.close()


