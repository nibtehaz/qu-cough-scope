import os
from flask import Flask, render_template, send_from_directory, request, jsonify, send_file
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
from datetime import datetime
from waitress import serve
import random
import string
import json
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import os
import h5py
import scipy.io as sio
import time
import pickle
import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import zipfile
from shutil import copyfile, rmtree
import librosa
import librosa.display
from Inception_Networks import *


# Data science tools
import numpy as np
import os
from skimage import io   
from skimage import measure, morphology
import matplotlib.pyplot as plt
import PIL
import cv2 

# Pytorch
import torch
from torchvision import transforms
from torch import optim, cuda, tensor
from torch.utils.data import DataLoader 
import torch.nn as nn

from DSP import classify_cough


################
# FLASK Params #
################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'database'


################
# Model Params #
################

###############
model_dir_1 = os.path.join('..','models','network1','densenet201_Cambridge_Cough_work_only_Breath_2Class_fold_1.pt')
model_dir_2  =  os.path.join('..','models','network2','inception_v3_Cambridge_Cough_work_only_Breath_2Class_fold_1.pt')
model_dir_3 =  os.path.join('..','models','network3','efficientnet_b7_Cambridge_Cough_work_only_Breath_2Class_fold_1.pt')
model_dir_4 =  os.path.join('..','models','network4','inception_v3_asymp_cough_fold_2.pt')
model_dir_5 =  os.path.join('..','models','network5','inception_v3_Cough_NewData_fold_1.pt')
model_dir_6 =  os.path.join('..','models','network6','efficientnet_b6_asymp_cough_fold_1.pt')
model_dir_7 =  os.path.join('..','models','network7','efficientnet_b7_asymp_cough_fold_2.pt')


device = 'cpu'    # 'cpu' or 'cuda' 
##############


#Load Model


checkpoint = torch.load(model_dir_1, map_location='cpu')
model_1 = checkpoint['model']  
checkpoint = torch.load(model_dir_2, map_location='cpu')
model_2 = checkpoint['model']  
checkpoint = torch.load(model_dir_3, map_location='cpu')
model_3 = checkpoint['model']  
checkpoint = torch.load(model_dir_4, map_location='cpu')
model_4 = checkpoint['model']  
checkpoint = torch.load(model_dir_5, map_location='cpu')
model_5 = checkpoint['model']  
checkpoint = torch.load(model_dir_6, map_location='cpu')
model_6 = checkpoint['model']  
checkpoint = torch.load(model_dir_7, map_location='cpu')
model_7 = checkpoint['model']  
del checkpoint 

# Set to evaluation mode
model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()
model_6.eval()
model_7.eval()

# set device to 'cpu' or 'cuda'
model_1 = model_1.to('cpu')  
model_2 = model_2.to('cpu')  
model_3 = model_3.to('cpu')  
model_4 = model_4.to('cpu')  
model_5 = model_5.to('cpu')  
model_6 = model_6.to('cpu')  
model_7 = model_7.to('cpu')  




input_mean = {
    'Symptomatic Breath' : [0.6973,0.6708,0.8114],
    'Symptomatic Cough' : [0.6795,0.6961,0.8669],
    'Asymp_Cough' : [0.6759,0.6911,0.8654]
}

input_std = {
    'Symptomatic Breath' : [0.3237,0.3119,0.1982],
    'Symptomatic Cough' : [0.3294,0.2932,0.1537],
    'Asymp_Cough' : [0.3329,0.2993,0.1512]
}


coughvid_model = pickle.load(open(os.path.join('coughvid_models', 'cough_classifier'), 'rb'))
coughvid_scaler = pickle.load(open(os.path.join('coughvid_models','cough_classification_scaler'), 'rb'))

def classify_cough_wrapper(cough_fl):

    x, sr = librosa.load(cough_fl)
    probability = classify_cough(x, sr, coughvid_model, coughvid_scaler)

    print(probability)
    

def predict(cough_fl, breath_fl, asymp):

    
    x, sr = librosa.load(cough_fl)
    
    X = librosa.stft(x) 
    Xdb = librosa.amplitude_to_db(abs(X)) 
    plt.figure(figsize=(14, 5)) 
    plt.axis('off') 
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log') 
    plt.savefig(cough_fl.split('.')[0]+'.png')

    
    x, sr = librosa.load(breath_fl)
    X = librosa.stft(x) 
    Xdb = librosa.amplitude_to_db(abs(X)) 
    plt.figure(figsize=(14, 5)) 
    plt.axis('off') 
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log') 
    plt.savefig(breath_fl.split('.')[0]+'.png')

    cough_img = cv2.imread('cough_fl'+'.png')
    cough_img = cv2.cvtColor(cough_img, cv2.COLOR_BGR2RGB) 
    breath_img = cv2.imread('breath_fl'+'.png')
    breath_img = cv2.cvtColor(breath_img, cv2.COLOR_BGR2RGB) 

    if (asymp) :
        
        cough_img[:,:,0] = (cough_img[:,:,0] - input_mean['Asymp_Cough'][0])/input_std['Asymp_Cough'][0]
        cough_img[:,:,1] = (cough_img[:,:,1] - input_mean['Asymp_Cough'][1])/input_std['Asymp_Cough'][1]
        cough_img[:,:,2] = (cough_img[:,:,2] - input_mean['Asymp_Cough'][2])/input_std['Asymp_Cough'][2]
        

        cough_img_rsz = cv2.resize(cough_img, (299,299))
        cough_img_rsz = np.array([cough_img_rsz[:,:,0],cough_img_rsz[:,:,1],cough_img_rsz[:,:,0]])
        x = torch.autograd.Variable(torch.Tensor([cough_img_rsz]))
        y5 = torch.exp(model_5(x)).detach().numpy()[0]

        cough_img_rsz = cv2.resize(cough_img, (224,224))
        cough_img_rsz = np.array([cough_img_rsz[:,:,0],cough_img_rsz[:,:,1],cough_img_rsz[:,:,0]])
        
        x = torch.autograd.Variable(torch.Tensor([cough_img_rsz]))
        
        y6 = torch.exp(model_6(x)).detach().numpy()[0]
        y7 = torch.exp(model_7(x)).detach().numpy()[0]
        
        E5 = y6[0]                   # in Tawsif's code 1st index is covid, 2nd index is healthy
        I5 = y7[0]                   # in Tawsif's code 1st index is covid, 2nd index is healthy
        G5 = y5[0]                   # in Tawsif's code 1st index is covid, 2nd index is healthy

        print(y5,y6,y7)

        if ((I5>0.95) or (I5>0.5 and E5>0.5 and G5>0.5) or (I5>0.5 and E5>0.5)or (E5>0.5 and G5>0.5) or (G5>0.5 and I5>0.5)):

            y = [0,1]

        else:
            y = [1,0]



    else:
        breath_img[:,:,0] = (breath_img[:,:,0] - input_mean['Symptomatic Breath'][0])/input_std['Symptomatic Breath'][0]
        breath_img[:,:,1] = (breath_img[:,:,1] - input_mean['Symptomatic Breath'][1])/input_std['Symptomatic Breath'][1]
        breath_img[:,:,2] = (breath_img[:,:,2] - input_mean['Symptomatic Breath'][2])/input_std['Symptomatic Breath'][2]

        breath_img_rsz = cv2.resize(breath_img, (224,224))
        breath_img_rsz = np.array([breath_img_rsz[:,:,0],breath_img_rsz[:,:,1],breath_img_rsz[:,:,0]])
        x = torch.autograd.Variable(torch.Tensor([breath_img_rsz]))
        y1 = torch.exp(model_1(x)).detach().numpy()[0]
        y3 = torch.exp(model_3(x)).detach().numpy()[0]

        breath_img_rsz = cv2.resize(breath_img, (299,299))
        breath_img_rsz = np.array([breath_img_rsz[:,:,0],breath_img_rsz[:,:,1],breath_img_rsz[:,:,0]])
        x = torch.autograd.Variable(torch.Tensor([breath_img_rsz]))
        y2 = torch.exp(model_2(x)).detach().numpy()[0]

        cough_img[:,:,0] = (cough_img[:,:,0] - input_mean['Symptomatic Cough'][0])/input_std['Symptomatic Cough'][0]
        cough_img[:,:,1] = (cough_img[:,:,1] - input_mean['Symptomatic Cough'][1])/input_std['Symptomatic Cough'][1]
        cough_img[:,:,2] = (cough_img[:,:,2] - input_mean['Symptomatic Cough'][2])/input_std['Symptomatic Cough'][2]

        cough_img_rsz = cv2.resize(cough_img, (299,299))
        cough_img_rsz = np.array([cough_img_rsz[:,:,0],cough_img_rsz[:,:,1],cough_img_rsz[:,:,0]])
        x = torch.autograd.Variable(torch.Tensor([cough_img_rsz]))
        y4 = torch.exp(model_4(x)).detach().numpy()[0]

        print(y1,y2,y3,y4)

        K4 = y4[0]                   # in Tawsif's code 1st index is covid, 2nd index is healthy
        H4 = y3[1]                   # in Tawsif's code 1st index is covid, 2nd index is healthy
        D4 = y2[1]                   # in Tawsif's code 1st index is covid, 2nd index is healthy
        F4 = y1[1]                   # in Tawsif's code 1st index is covid, 2nd index is healthy

        if((K4>0.9) or (H4>0.99) or (D4>0.5 and F4>0.5 and H4>0.7) or (D4>0.5 and F4>0.5) or (F4>0.5 and H4>0.7) or (H4>0.7 and D4>0.5)):
            y = [1,0]
        else:
            y = [0,1]


    
    os.remove(cough_fl.split('.')[0]+'.png')
    os.remove(breath_fl.split('.')[0]+'.png')

    return y


@app.route('/', methods = ['GET', 'POST'])
def home():
	
    return render_template("index.html")

@app.route('/web', methods = ['GET', 'POST'])
def web():
	
    return render_template("index_web.html")



@app.route('/negative', methods = ['GET', 'POST'])
def negative():
	
    return render_template("negative.html")


@app.route('/positive', methods = ['GET', 'POST'])
def positive():
	
    return render_template("positive.html")


@app.route('/negative_arabic', methods = ['GET', 'POST'])
def negative_arabic():
	
    return render_template("negative_arabic.html")


@app.route('/positive_arabic', methods = ['GET', 'POST'])
def positive_arabic():
	
    return render_template("positive_arabic.html")




def create_user_id():

    return str(datetime.now().strftime('%Y%m%d%H%M%S%f'))+ ''.join(random.choice(string.ascii_uppercase + string.digits + string.ascii_lowercase) for _ in range(5))

@app.route('/cough_upload', methods = ['POST'])
def cough_upload():    

    today = str(datetime.now())
    day = secure_filename(today.split(' ')[0])
    time = secure_filename( '-'.join(today.split(' ')[1].split('.')) )

    try:
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'symptom_data', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'user_data', day))
    except:
        pass



    user_data = json.loads(request.form['userdata'])
    symptom_data = json.loads(request.form['symptomdata'])

    if 'userid' not in user_data:
        user_id = create_user_id()
        user_data['userid'] = user_id
        
        fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'user_data', day,user_data['userid']+'.json'), 'w')
        fp.write(str(user_data))
        fp.close()


    fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'symptom_data', day,user_data['userid']+'_'+time+'.json'), 'w')
    fp.write(str(symptom_data))
    fp.close()


    cough_content = request.files['cough_data'].read()
    fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day,user_data['userid']+'_'+time+'.wav'), 'wb')
    fp.write(cough_content)
    fp.close()


    breath_content = request.files['breath_data'].read()
    fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day,user_data['userid']+'_'+time+'.wav'), 'wb')
    fp.write(breath_content)
    fp.close()

    #classify_cough_wrapper(os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day,user_data['userid']+'_'+time+'.wav'))
    #classify_cough_wrapper(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day,user_data['userid']+'_'+time+'.wav'))


    asymp = True

    for symptom in symptom_data['symptoms']:

        if 'cough' in symptom:
            asymp = False

    y = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day,user_data['userid']+'_'+time+'.wav'),
                os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day,user_data['userid']+'_'+time+'.wav'),
                asymp)

    if (y[0]<y[1]):
        predicted_class = 'covid'
    else:
        predicted_class = 'normal'
    

    print('----------------------')
    print(day,time)
    print(f"{'Asymptomatic' if asymp else 'Symptomatic'} patient is predicted {predicted_class}")
    print(y)
    print('----------------------')

    return jsonify({'predicted_class':predicted_class,'userid':user_data['userid']})



@app.route('/audio_upload', methods = ['POST'])
def audio_upload():    

    today = str(datetime.now())
    day = secure_filename(today.split(' ')[0])
    time = secure_filename( '-'.join(today.split(' ')[1].split('.')) )

    try:
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'symptom_data', day))
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'user_data', day))
    except:
        pass



    user_data = json.loads(request.form['userdata'])
    symptom_data = json.loads(request.form['symptomdata'])

    if 'userid' not in user_data:
        user_id = create_user_id()
        user_data['userid'] = user_id
        
        fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'user_data', day,user_data['userid']+'.json'), 'w')
        fp.write(str(user_data))
        fp.close()


    fp = open(os.path.join(app.config['UPLOAD_FOLDER'], 'symptom_data', day,user_data['userid']+'_'+time+'.json'), 'w')
    fp.write(str(symptom_data))
    fp.close()

    cough_file = request.files.getlist("cough")[0]
    cough_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day,user_data['userid']+'_'+time+'.wav'))

    breath_file = request.files.getlist("breath")[0]
    breath_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day,user_data['userid']+'_'+time+'.wav'))
   

    asymp = True

    for symptom in symptom_data['symptoms']:

        if 'cough' in symptom:
            asymp = False

    y = predict(os.path.join(app.config['UPLOAD_FOLDER'], 'breath_sound', day,user_data['userid']+'_'+time+'.wav'),
                os.path.join(app.config['UPLOAD_FOLDER'], 'cough_sound', day,user_data['userid']+'_'+time+'.wav'),
                asymp)

    if (y[0]<y[1]):
        predicted_class = 'covid'
    else:
        predicted_class = 'normal'
    

    print('----------------------')
    print(day,time)
    print(f"{'Asymptomatic' if asymp else 'Symptomatic'} patient is predicted {predicted_class}")
    print(y)
    print('----------------------')

    return jsonify({'predicted_class':predicted_class,'userid':user_data['userid']})



if __name__ == '__main__':

    
	# running the app

    app.debug = True
	#app.run(host="127.0.0.1",port="5000")
    serve(app, host='0.0.0.0', port=5000, threads=8) #WAITRESS!
