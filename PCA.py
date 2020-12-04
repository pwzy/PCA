import torchvision.models as models
from torchvision import transforms
import os
from PIL import Image
from sklearn.decomposition import PCA
import numpy as np
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model = models.resnet18(pretrained=True).to(device)

feature = []

for root, dirs, files in os.walk('./ped2/testing/frames/01/'):
    for file in files:
        img_path = os.path.join(root, file)
        # 读入图片
        img = Image.open(img_path)
        tf = transforms.ToTensor()
        
        img_tensor = tf(img).unsqueeze(0).to(device)
        
        feature_extract = model(img_tensor).squeeze().detach().cpu().numpy()

        feature.append(feature_extract)


data = np.array(feature)

#  print(data)
#  print(type(data))
#  print(len(feature))
#  print(feature[0].shape)

pca = PCA(n_components=3)
reduced_data = pca.fit_transform(data)

#  print(reduced_data)

np.save('feature', reduced_data)


     


