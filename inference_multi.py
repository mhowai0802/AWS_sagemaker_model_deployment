# Requirement.txt
import os
os.system("pip install logging torch torchvision sklearn PIL io")
from os.path import isfile, join
from os import listdir
# Require# Requirement.txt
import os
os.system("pip install logging torch torchvision sklearn PIL io")
from os.path import isfile, join
from os import listdir
# Required packages to import
import logging
import torch
import torch.utils.data
from sklearn.metrics import confusion_matrix
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import requests
from PIL import Image
from io import BytesIO
import json

dic = {0: 'Blepharitis + Conjunctivitis',
       1: 'Cataract',
       2: 'Cherry Eye',
       3: 'Corneal Edema + Corneal Ulceration',
       4: 'Glaucoma',
       5: 'Hyphema + Uveitis',
       6: 'KCS',
       7: 'Nuclear Sclerosis'}


class NN_Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

class NN_Classifier_binary(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.1):
        super().__init__()

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = self.output(x)

        return F.sigmoid(x).squeeze(1)    
    

def input_fn(request_body, content_type='application/json'):
    # get the image from URL and do transformation
    response = requests.get(request_body)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    transform = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(244),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
    testloader = torch.utils.data.DataLoader(img_tensor, batch_size=1)
    
    return testloader    

def model_fn(model_dir):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = os.listdir(".")
    model = models.densenet201(pretrained=False)
    n_in = next(model.classifier.modules()).in_features
    n_hidden = [1024, 512, 256]
    n_out = 8  # 8 classes
    model.classifier = NN_Classifier(input_size=n_in, output_size=n_out, hidden_layers=n_hidden, drop_p=0)
    logging.info(f'Model_directory: {model_dir}')
    logging.info(f'File_inside:{x}')
    path = os.path.join(model_dir, 'unfreeze2_8_0712_0837_0894.pt')
    logging.info(f'Model_file:{path}')
    with open(path, 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model

def predict_fn(testloader, model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    for images in testloader:
        images = images.to(device)
        output = model(images)
        ps = torch.exp(output)
    res = ps
    top3 = res.topk(3, dim=1)[1]
    top3_percent = res.topk(3, dim=1)[0]
    top3_percent_list = top3_percent.tolist()[0]
    dict_master = {}
    for i in range(len(top3_percent_list)):
        dict_master[f'{i + 1}.{dic[top3[0][i].item()]}'] = f'{top3_percent_list[i] / sum(top3_percent_list)}'
    return json.dumps(dict_master,indent=4)

def output_fn(prediction, accept='application/json'):
    return prediction
