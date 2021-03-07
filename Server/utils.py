# Printing out all outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
# PyTorch
import torch
from torchvision import transforms, models
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
import numpy as np
import pandas as pd
import os
from skimage import io
# Image manipulations
from PIL import Image
# Timing utility
from timeit import default_timer as timer
# Visualizations
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

class Identity(nn.Module):
    def _init_(self):
        super(Identity, self)._init_()
        
    def forward(self, x):
        return x
        
class CustomLayer(nn.Module):
            def __init__(self,layer_idx=None,in_channels=1,out_channels=1,kernel_size=1,sampling_factor=1,optimize=True):
                super().__init__() 
                self.in_channels = in_channels 
                self.out_channels = out_channels
                self.kernel_size = kernel_size
                self.sampling_factor = sampling_factor
                self.layer_idx = layer_idx
            def forward(self,x): 
                x = x.squeeze() 
                return x


class MyData(Dataset):
    
    def __init__(self, root_dir,categories,img_names,target,my_transforms, return_path=False):
        self.root_dir = root_dir
        self.categories = categories
        self.img_names = img_names
        self.target = target
        self.my_transforms = my_transforms
        self.return_path = return_path

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, index):
        # read image
        y = self.target[index].squeeze()
        label = y.item()
        x = io.imread(os.path.join(self.root_dir, self.categories[label]+'/'+self.img_names[index]))
        # apply transformation
        x = self.my_transforms(x) 
        # # normalize per channel
        # # work for gray images only
        # x = (x-x[0,:,:].mean())/x[0,:,:].std()  
        # # work both for gray coloured images  
        # x_mean = x.mean(dim=(1,2)) 
        # x_std = x.std(dim=(1,2)) 
        # x = (x-x_mean.unsqueeze(1).unsqueeze(2)) / x_std.unsqueeze(1).unsqueeze(2)
        if self.return_path: 
            return x, y, self.img_names[index]
        else:
            return x, y




def get_pretrained_model(parentdir, model_name,class_num,train_on_gpu,multi_gpu):
    """Retrieve a pre-trained model from torchvision

    Params
    -------
        model_name (str): name of the model (currently only accepts vgg16 and resnet50)

    Return
    --------
        model (PyTorch model): cnn

    """

    if model_name == 'squeezenet1_0':
        # model = models.squeezenet1_0(pretrained=True) 
        from squeezenet import squeezenet1_0
        model = squeezenet1_0(pretrained=True) 
        # class CustomLayer(nn.Module):
        #     def __init__(self,layer_idx=None,in_channels=1,out_channels=1,kernel_size=1,sampling_factor=1,optimize=True):
        #         super().__init__() 
        #         self.in_channels = in_channels 
        #         self.out_channels = out_channels
        #         self.kernel_size = kernel_size
        #         self.sampling_factor = sampling_factor
        #         self.layer_idx = layer_idx
        #     def forward(self,x): 
        #         x = x.squeeze() 
        #         return x 
        model.classifier[-1] = nn.Sequential(  
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            CustomLayer(), 
            nn.Linear(1000, 256), nn.ReLU(), nn.Dropout(0.2), 
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)  
            )   
        # model = nn.Sequential(model, 
        #     nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Dropout(0.2), 
        #     nn.Linear(256, class_num), nn.LogSoftmax(dim=1)) 
        #     )   
    
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True) 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        # # Freeze early layers 
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        n_inputs = model.classifier[-1].in_features
        # Add on classifier
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier  
        n_inputs = model.fc.in_features 
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), 
            nn.LogSoftmax(dim=1)
            # nn.Softmax(dim=1)  

            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.Softmax(dim=1) 
            )

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

    elif model_name == 'inception_v3':
        from Inception_Networks import inception_v3
        model = inception_v3(pretrained=True) 

        # InceptionV3 = torch.load('models/inception_v3.pth')
        # model  = InceptionV3['model']
        # del InceptionV3    

        # import pretrainedmodels 
        # import ssl
        # ssl._create_default_https_context = ssl._create_unverified_context
        # model = pretrainedmodels.__dict__['inceptionv3'](num_classes=1000, pretrained='imagenet')
        # ssl._create_default_https_context = ssl._create_stdlib_context 

        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )

        # model = nn.Sequential(model, 
        #     nn.Sequential(nn.Linear(1000, 256), nn.ReLU(), nn.Dropout(0.2), 
        #     nn.Linear(256, class_num), nn.LogSoftmax(dim=1)) 
        #     )  

    elif model_name == 'inceptionresnetv2':
        from inceptionresnetv2 import inceptionresnetv2
        model = inceptionresnetv2(parentdir, num_classes=1000, pretrained='imagenet')
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False 
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    elif model_name == 'xception':
        from xception import xception
        model = xception(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    elif model_name == 'chexnet':
        from chexnet import chexnet
        model = chexnet(parentdir)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.module.densenet121.classifier[0].in_features
        model.module.densenet121.classifier = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  
       
    if model_name == 'nasnetalarge':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    if model_name == 'pnasnet5large':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )   

    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    if model_name == 'densenet201': 
        model = models.densenet201(pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    if model_name == 'shufflenet':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    if model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  


    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )  

    if model_name == 'nasnetamobile':
        import pretrainedmodels
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        ssl._create_default_https_context = ssl._create_stdlib_context 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier
        n_inputs = model.last_linear.in_features
        model.last_linear = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            )   

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # model = torch.hub.load('pytorch/vision:v0.6.0', 'shufflenet_v2_x1_0', pretrained=True)
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            ) 

    if model_name == 'darknet53':
        from darknet53 import darknet53 
        model = darknet53(1000)
        checkpoint = parentdir + 'models/darknet53.pth.tar'
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint ['state_dict'])
        del checkpoint 
        # # Freeze early layers
        # for param in model.parameters():
        #     param.requires_grad = False
        # Add on classifier 
        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, class_num), nn.LogSoftmax(dim=1)
            # nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1)
            ) 
    elif model_name == 'efficientnet_b0':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b0') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)
        # model = nn.Sequential(*list(model.children())[:-1]) 

    elif model_name == 'efficientnet_b1':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b1') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b2':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b2') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b3':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b3') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b4':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b4') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b5':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b5') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b6':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b6') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

    elif model_name == 'efficientnet_b7':
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7') 
        n_inputs = model._fc.in_features 
        model._fc = nn.Sequential(
            nn.Linear(n_inputs, class_num), nn.LogSoftmax(dim=1) 
            ) 
        model._swish =  Identity() 
        torch.nn.init.xavier_uniform_(model._fc[0].weight)
        model._fc[0].bias.data.fill_(0.01)

  


    # Move to gpu and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)

    return model 

 
def Createlabels(datadir):
    categories = []
    n_Class = []
    img_names = []
    labels = []
    i = 0
    class_to_idx = {}
    idx_to_class = {}
    for d in os.listdir(datadir): 
        class_to_idx[d] = i
        idx_to_class[i] = d  
        categories.append(d)
        temp = os.listdir(datadir + d)
        img_names.extend(temp)
        n_temp = len(temp)
        if i==0:
            labels = np.zeros((n_temp,1)) 
        else:
            labels = np.concatenate( (labels, i*np.ones((n_temp,1))) )
        i = i+1
        n_Class.append(n_temp)

    return categories, n_Class, img_names, labels,  class_to_idx, idx_to_class
    

def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data 
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def train(model_to_load,
          model,
          criterion,
          optimizer,
          scheduler,
          train_loader,
          valid_loader,
          save_file_name,
          train_on_gpu,
          history=[],
          max_epochs_stop=5,
          n_epochs=30,
          print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')

    overall_start = timer()

    # Main loop
    for epoch in range(n_epochs):

        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(train_loader):
            # Tensors to gpu
            if train_on_gpu:
                data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)

            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            # 'using inception_v3' 
            output = model(data)    
            if model_to_load=='inception_v3':
                output = output[0] 
            

            # Loss and backpropagation of gradients 
            # loss = criterion(output, to_one_hot(target).to('cuda'))  # use it with mse loss 
            loss = criterion(output, target) 
            loss.backward() 

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)
            
            # Track training progress
            # print(
            #     f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #     end='\r')

            # release memeory (delete variables)
            del output, data, target 
            del loss, accuracy, pred, correct_tensor 

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in valid_loader:
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.to('cuda', non_blocking=True), target.to('cuda', non_blocking=True)
                        # data, target = data.cuda(), target.cuda()

                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)
                # scheduler.step(valid_loss) 

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    )

                # release memeory (delete variables)
                del output, data, target
                del loss, accuracy, pred, correct_tensor 

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                # if 1:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_best_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_loss', 'valid_loss', 'train_acc',
                                'valid_acc'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history






