import os
import torch
from RFDN.RFDN import RFDN
from RFDN.utils import utils_image as util
from tqdm import tqdm

# --------------------------------
# load model
# --------------------------------
model_path = os.path.join('RFDN/trained_model', 'RFDN_AIM.pth')
model = RFDN()
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for k, v in model.named_parameters():
    v.requires_grad = False
model = model.to("cuda")

# --------------------------------
# read image
# --------------------------------
L_folder = os.path.join("../prid2011/prid_2011/multi_shot")
img_paths = []

for cam in os.listdir(L_folder):
    cam_path = os.path.join(L_folder, cam)
    for person in os.listdir(cam_path):
        person_path = os.path.join(cam_path, person)
        for name in os.listdir(person_path):
            img_path = os.path.join(person_path, name)
            img_paths = img_paths.append(img_path)

for img in tqdm(img_paths):
    # --------------------------------
    # (1) img_L
    # --------------------------------
    img_L = util.imread_uint(img, n_channels=3)
    img_L = util.uint2tensor4(img_L)
    img_L = img_L.to("cuda")

    img_E = model(img_L)
    torch.cuda.synchronize()

    # --------------------------------
    # (2) img_E
    # --------------------------------
    img_E = util.tensor2uint(img_E)

    # --------------------------------
    # (3) save results
    # --------------------------------
    util.imsave(img_E, "../sr/" + img.replace("/", "*"))
