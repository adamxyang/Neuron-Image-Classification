import os, torch
import numpy as np
import PIL
import cv2
import pandas as pd
from sklearn.utils import shuffle
from torchvision import models
import torch.nn as nn
import time
from utils_old import train, evaluate, calculate_accuracy
from tqdm import tqdm
from cvtorchvision import cvtransforms
import pickle
from itertools import groupby


def open_1_channel(fname, stain):
    file_list = fname.split('/separator/')
    if stain == 'cy5':
        s = 0
    elif stain == 'dapi':
        s = 1
    elif stain == 'dsred':
        s = 2
    elif stain == 'fitc':
        s = 3
    x = cv2.imread(file_list[s], cv2.IMREAD_GRAYSCALE)
    x = np.expand_dims(x, axis=0)
    return torch.tensor(x, dtype=torch.float)


class my_one_dataset(object):
    def __init__(self, df, stain, transform=None, inference=False, gradcam=False):
        self.df = df
        self.transform = transform
        self.inference = inference
        self.gradcam = gradcam
        self.stain = stain

    def __getitem__(self, i):
        fname = self.df.all_files.tolist()[i]
        x = open_1_channel(fname, self.stain)
        if self.transform:
            x = self.transform(images=x)
        if self.inference:
            y = self.df.treatment.tolist()[i]
            row = self.df.row.tolist()[i]
            col = self.df.col.tolist()[i]
            fld = self.df.fld.tolist()[i]
            drug = self.df.drug.tolist()[i]
            return x,y,row,col,fname,fld,drug
        elif self.gradcam:
            row = self.df.row.tolist()[i]
            col = self.df.col.tolist()[i]
            fld = self.df.fld.tolist()[i]
            drug = self.df.drug.tolist()[i]
            drug_dose = self.df.drug_dose.tolist()[i]
            AB_dose = self.df.AB_dose.tolist()[i]
            prob = self.df.prob.tolist()[i]
            return x,row,col,fname,fld,drug, drug_dose, AB_dose, prob
        else:
            y = self.df.treatment_int.tolist()[i]
            y = torch.tensor(y)
            return x,y

    def __len__(self):
        return len(self.df)

    
import imgaug as ia
import imgaug.augmenters as iaa

def get_loader(drug_list, stain):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential([
#             iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.Flipud(0.2), # vertically flip 20% of all images
#             sometimes(iaa.Affine(
#                 scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#                 translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#                 rotate=(-45, 45), # rotate by -45 to +45 degrees
# #                 shear=(-16, 16), # shear by -16 to +16 degrees
# #                 order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
# #                 cval=(0, 255), # if mode is constant, use a cval between 0 and 255
# #                 mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#             )),
#             iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
            ])
    
    df_new = pd.read_csv('/mnt/sdh/adam/test/df_new.csv')
    df_new['treatment_int'] = pd.factorize(df_new.treatment)[0]   # ab=0, veh=1
    df_new['all_files'] = df_new.apply(lambda row: row.cy5_file + '/separator/' + row.dapi_file + '/separator/' + row.dsred_file + '/separator/' + row.fitc_file, axis=1)
    df_new = df_new[df_new.drug_dose==0]
    valid_df = df_new[df_new.drug.isin(drug_list)]
    train_df = df_new.drop(valid_df.index)
    train_df = shuffle(train_df)
    valid_df = shuffle(valid_df)
    train_set = my_one_dataset(train_df, stain, transform=None)
    valid_set = my_one_dataset(valid_df, stain, transform=None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2)
    return train_loader, valid_loader


def get_model():
    model = models.resnet34(pretrained=True)
    
    kernel = model.conv1.weight
    new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:,:] = torch.stack([torch.mean(kernel, 1)], dim=1)
    model.conv1 = new_conv
    
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    return model


def train_1(valid_drug_list, stain, model_pth=None, gpu='6'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device('cuda:0')
    model = get_model()
    model.to(device);
    
    train_loader, valid_loader = get_loader(valid_drug_list, stain=stain)
    if model_pth:
        learning_rate = 1e-4
        model.load_state_dict(torch.load(model_pth))
        print('Unfreezing all layers...')
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = True
    else:
        model_pth = '/mnt/sdh/adam/test/models/' + stain + '_model.pth'
        print('Unfreezing fc and layer4...')
        learning_rate = 1e-3
        for name, child in model.named_children():
            if name in ['fc','layer4']:
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False
    model.to(device);

    epochs = 10
    best_valid_loss = float('inf')
    criterion = torch.nn.CrossEntropyLoss()
    momentum = 0.9
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                    momentum=momentum)
    for epoch in range(epochs):
        tic = time.time()
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, device, valid_loader, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_pth)
            print('Best epoch: 0%d' %(epoch+1))
        toc = time.time()

        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:05.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:05.2f}% | Time: {toc-tic:.2f}s')

        
        
def get_prob_list(drug_list, model_name='cy5', model_pth='/mnt/sdh/adam/test/models/cy5_model.pth', gpu='6'):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device('cuda:0')
    model = get_model()
    model.to(device)
    model.load_state_dict(torch.load(model_pth))
    df_new = pd.read_csv('/mnt/sdh/adam/test/df_new.csv')
    df_new['all_files'] = df_new.apply(lambda row: row.cy5_file + '/separator/' + row.dapi_file + '/separator/' + row.dsred_file + '/separator/' + row.fitc_file, axis=1)
    for drug in drug_list:
        for drug_dose in [0,1,3,10]:
            for treatment_dose in [0,30]:
                print('predicting for compound %s, dose %d, AB dose %d' %(drug, drug_dose, treatment_dose))
                df_new_selected = df_new[(df_new.drug_dose==drug_dose) & (df_new.treatment_dose==treatment_dose) & 
                            (df_new.drug==drug)]
                valid_set = my_one_dataset(df_new_selected, transform=None, stain=model_name, inference=True)
                valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1)
                tic = time.time()
                prob_list = []
                with torch.no_grad():
                    for x,y,row,col,fname,fld,drug_valid in valid_loader:
                        x = x.to(device, dtype=torch.float)
                        fx = model(x)
                        prob = torch.nn.Softmax(dim=1)(fx)
                        prob_list.append((float(prob[0][0]),int(row),str(col[0]),fname[0],str(int(fld)),str(drug)))
                with open('./prob_list/'+model_name+drug+str(drug_dose)+'_'+str(treatment_dose)+'.txt', "wb") as fp:
                    pickle.dump(prob_list, fp)
                toc = time.time()
                print('%d samples, time taken %ds' %(len(prob_list),toc-tic))

                
def one_well_df(drug, model_name):
    df = pd.DataFrame(index=range(30))
    for d in [0,1,3,10]:
        for t in [0,30]:
            keys = []
            with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                mylist = pickle.load(fp)
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
                    keys.append(key)
                for i in keys:
                    if not i.endswith('_file'):
                        keys.append(i+'_file')
                df = df.join(pd.DataFrame(columns=keys, index=range(30)))  # join a new drug config
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])): # x[1]+x[2] = well position (6B)
                    for thing in group:
                        prob = thing[0]
                        file = thing[3]
                        df[key] = pd.Series([x for x in df[key].tolist() if str(x) != 'nan']+[prob])
                        df[key+'_file'] = pd.Series([x for x in df[key+'_file'].tolist() if str(x) != 'nan']+[file])
    return df


def one_config_df(drug, drug_dose, treatment_dose, model_name):
    d = drug_dose
    t = treatment_dose
    keys = []
    with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
        mylist = pickle.load(fp)
        for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
            keys.append(key)
        df = pd.DataFrame(columns=keys, index=range(30))
        for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])): # x[1]+x[2] = well position (6B)
            for thing in group:
                prob = thing[0]
                df[key] = pd.Series([x for x in df[key].tolist() if str(x) != 'nan']+[prob])
    return df



def out_df(drug_list, model_name):
    num = len(drug_list)
    df = pd.DataFrame(columns=['compound', 'dose', 'AB', 'well_1', 'well_2', 'well_3', 'well_4', 'well_5', 'well_6',
    'similarity_to_AB', 'avg_well_std'])
    for drug in drug_list:
        for d in [0,1,3,10]:
            for t in [0,30]:
                one_df = one_config_df(drug, d, t, model_name)
                cols = one_df.columns
                avg_prob = []
                std = []
                for i in range(6):
                    avg_prob.append(np.mean(one_df[cols[i]].tolist()))
                    std.append(np.std(one_df[cols[i]].tolist()))
                total_avg = np.mean(avg_prob)
                avg_prob = np.around(avg_prob, decimals=2)
                total_avg = np.around(total_avg, decimals=2)
                avg_std = np.around(np.mean(std), decimals=2)
                df = df.append({'compound':drug, 'dose':d, 'AB':t, 'well_1':avg_prob[0], 'well_2':avg_prob[1], 'well_3':avg_prob[2], 'well_4':avg_prob[3], 'well_5':avg_prob[4], 'well_6':avg_prob[5], 'similarity_to_AB':total_avg, 'avg_well_std':avg_std}, ignore_index=True)
    return df



def well_df(drug, file_path = '/mnt/sdh/adam/test/prob_lists/', model_name='cy5'):
    df = pd.DataFrame(index=range(30))
    for d in [1,3,10]:
        for t in [0,30]:
            keys = []
            with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                mylist = pickle.load(fp)
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
                    keys.append(key)
                for i in keys:
                    if not i.endswith('_file'):
                        keys.append(i+'_file')
                df = df.join(pd.DataFrame(columns=keys, index=range(30)))
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
                    for thing in group:
                        prob = thing[0]
                        file = thing[3]
                        df[key] = pd.Series([x for x in df[key].tolist() if str(x) != 'nan']+[prob])
                        df[key+'_file'] = pd.Series([x for x in df[key+'_file'].tolist() if str(x) != 'nan']+[file])
    return df


import matplotlib.pyplot as plt
from gradcam_new import *
import matplotlib as mpl
from matplotlib.pyplot import savefig

def plot_gradcam(drug_list, num, model_pth='./models/cy5_model.pth', top_losses=True, model_name='cy5'):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    device = torch.device('cuda:0')
    mpl.rcParams.update(mpl.rcParamsDefault)
    mylist = []
    for drug in drug_list:
        for d in [1,3,10]:
            for t in [0,30]:
                with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                    new_list = pickle.load(fp)
                    new_list1 = [x + (d,t) for x in new_list]
                    if t == 30:
                        def change_one(x):
                            l = list(x)
                            l[0] = 1-l[0]
                            return tuple(l)
                        new_list1 = [change_one(x) for x in new_list1]
                    mylist += new_list1
    sorted_list = sorted(mylist, key=lambda x: x[0], reverse=False)  # sort in descending order (True)
    sorted_list = sorted_list[:num//2]
    df = pd.DataFrame(columns=['prob','all_files','row','col','fld','drug','drug_dose', 'AB_dose'])
    for x in sorted_list:
        df = df.append({'prob':x[0], 'all_files':x[3], 'row':x[1], 'col':x[2], 'fld':x[4], 'drug':x[5], 'drug_dose':x[6], 'AB_dose':x[7]}, ignore_index=True)
    valid_df = df
    valid_set = my_one_dataset(valid_df, transform=None, gradcam=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1)
    
    model = get_model()
    model.load_state_dict(torch.load(model_pth))
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
            
    fig = plt.figure(figsize=(20,5*num//4))
    i = 0
    for x,row,col,fname,fld,drug, drug_dose, AB_dose, label_prob in valid_loader:
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            fx = model(x)
            prob = torch.nn.Softmax(dim=1)(fx)
            y_pred = int(np.argmax(prob.cpu()))
            y_prob = float(prob[0][y_pred])
            y_pred = 'AB' if y_pred==0 else 'VEH'
        grad_cam = GradCAM(model, 'layer4');
        mask = grad_cam(x,None);
        mask = cv2.resize(mask, (2048, 2048));
        plt.subplot(num//4,4,i+1)
        plt.imshow(x.squeeze().squeeze())  #.permute(1,2,0)[:,:,:])
        plt.title('drug %s %d, AB %d' %(str(drug[0]), drug_dose, AB_dose))
        plt.subplot(num//4,4,i+2)
        plt.imshow(x.squeeze().squeeze())
        plt.imshow(mask, alpha=0.6, cmap='magma')
        plt.title('prob wrong %.2f, row %s col %s fld %s' %(label_prob, str(np.array(row)[0]), str(col[0]), str(fld[0]).replace('tensor','').replace('([','').replace('])','')))
        i += 2
    plt.show()
    
    
    
def save_gradcam(drug_list, model_pth='./models/cy5_model.pth', model_name='cy5'):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="6"
    device = torch.device('cuda:0')
    mpl.rcParams.update(mpl.rcParamsDefault)
    mylist = []
    for drug in drug_list:
        for d in [1,3,10]:
            for t in [0,30]:
                with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                    new_list = pickle.load(fp)
                    new_list1 = [x + (d,t) for x in new_list]
                    if t == 0:
                        def change_one(x):
                            l = list(x)
                            l[0] = 1-l[0]
                            return tuple(l)
                        new_list1 = [change_one(x) for x in new_list1]
                    mylist += new_list1
    mylist_new = []
    for drug in drug_list:
        for d in [1,3,10]:
            for t in [0,30]:
                for x in mylist:
                    if (x[5],x[6],x[7]) == (drug,d,t):
                        break
                mylist_new.append(x)
                
    df = pd.DataFrame(columns=['prob','all_files','row','col','fld','drug','drug_dose', 'AB_dose'])
    for x in mylist_new:
        df = df.append({'prob':x[0], 'all_files':x[3], 'row':x[1], 'col':x[2], 'fld':x[4], 'drug':x[5], 'drug_dose':x[6], 'AB_dose':x[7]}, ignore_index=True)
    valid_df = df
    valid_set = my_one_dataset(valid_df, transform=None, gradcam=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1)
    
    model = get_model()
    model.load_state_dict(torch.load(model_pth))
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
            
    for x,row,col,fname,fld,drug, drug_dose, AB_dose, true_prob in valid_loader:
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            fx = model(x)
            prob = torch.nn.Softmax(dim=1)(fx)
            y_pred = int(np.argmax(prob.cpu()))
            y_prob = float(prob[0][y_pred])
            y_pred = 'AB' if y_pred==0 else 'VEH'
           
        
        fig = plt.figure(figsize=(20,20))
        plt.imshow(x.squeeze().squeeze())
        plt.axis('off')
        savefig('./figures/drug_%s_%d-AB_%d-original.jpg' %(str(drug[0]), drug_dose, AB_dose), bbox_inches='tight')
        plt.show()
        plt.close(fig)
        
        for index in [0,1]:
            grad_cam = GradCAM(model, 'layer4');
            mask = grad_cam(x,index=index);
            mask = cv2.resize(mask, (2048, 2048));
            if index == 0:
                grad_label = 'AB'
            else: 
                grad_label = 'VEH'

            fig = plt.figure(figsize=(20,20))
            plt.imshow(x.squeeze().squeeze())
            plt.imshow(mask, alpha=0.6, cmap='magma')
            plt.axis('off')
            savefig('./figures/drug_%s_%d-AB_%d-prob_%.3f-gradient_%s.jpg' %(str(drug[0]), drug_dose, AB_dose, true_prob, grad_label), bbox_inches='tight')
            plt.show()
            plt.close(fig)
        
import seaborn as sns

def plot_all_well(drug, file_path = './prob_lists/', model_name='cy5'):
    sns.set(rc={'figure.figsize':(5*6,5*6)})
    fig=plt.figure()
    row = 0
    for d in [1,3,10]:
        for t in [0,30]:
            keys = []
            with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                mylist = pickle.load(fp)
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
                    keys.append(key)
                df = pd.DataFrame(columns=keys, index=range(30))
                for key, group in groupby(mylist, lambda x: str(x[1])+str(x[2])):
                    for thing in group:
                        prob = thing[0]
                        df[key] = pd.Series([x for x in df[key].tolist() if str(x) != 'nan']+[prob])
            for key,pos in zip(keys,range(6)):  # colnames
                ax = fig.add_subplot(6,6,pos+1+row*6)
                ax.set(xlim=(0,1))
                ax.set(ylim=(0,40))
                sns.distplot(df[key].tolist(), ax=ax, bins=10)
                ax.set(title=r'%s %d %d %s' %(drug,d,t,key))
            row += 1
    savefig('./figures/drug_%s_distribution.jpg' %(drug), bbox_inches='tight')
    plt.show()


from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
def plot_calibration(drug_list, file_path = './prob_lists/', model_name='cy5'):
    sns.set(rc={'figure.figsize':(5,5)})
    mylist = []
    for drug in drug_list:
        d = 0
        for t in [0,30]:
            with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                new_list = pickle.load(fp)
                new_list1 = [x + (d,t) for x in new_list]
                if t == 0:
                    new_list1 = [x + (0,) for x in new_list]
                elif t == 30:
                    new_list1 = [x + (1,) for x in new_list]
                mylist += new_list1
    pred_prob = [x[0] for x in mylist]
    label = [x[-1] for x in mylist]
    x,y = calibration_curve(label, pred_prob, n_bins=10)
    
    fig = plt.figure(figsize=(20,20))
    plt.plot(x,y, marker='o', linewidth=1)
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.plot([0.0, 1.0], [0.0, 1.0], color='black')
    savefig('./figures/calibration.jpg', bbox_inches='tight')
    plt.show()
    
    
    
def plot_calibration_comparison(drug_list, file_path = './prob_lists/', model_name_list=[('cy5','original'),
                                                                                         ('cy5_temp','tuned')]):
    sns.set(rc={'figure.figsize':(20,20)})
    fig = plt.figure(figsize=(10,10))
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.plot([0.0, 1.0], [0.0, 1.0], color='black')
    
    for (model_name,plot_label) in model_name_list:
        mylist = []
        for drug in drug_list:
            d = 0
            for t in [0,30]:
                with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                    new_list = pickle.load(fp)
                    new_list1 = [x + (d,t) for x in new_list]
                    if t == 0:
                        new_list1 = [x + (0,) for x in new_list]
                    elif t == 30:
                        new_list1 = [x + (1,) for x in new_list]
                    mylist += new_list1
        pred_prob = [x[0] for x in mylist]
        label = [x[-1] for x in mylist]
        x,y = calibration_curve(label, pred_prob, n_bins=10)
        plt.plot(x,y, marker='o', linewidth=1, label=plot_label)
    plt.legend(loc='lower right')
    savefig('./figures/calibration_comparison.jpg', bbox_inches='tight')
    plt.show()
    
    
    
def plot_gradcam_unsure(drug_list, num, model_pth='./models/cy5_model.pth', top_losses=True, model_name='cy5'):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="7"
    device = torch.device('cuda:0')
    mpl.rcParams.update(mpl.rcParamsDefault)
    mylist = []
    for drug in drug_list:
        for d in [1,3,10]:
            for t in [0,30]:
                with open('./prob_list/'+model_name+drug+str(d)+'_'+str(t)+'.txt', "rb") as fp:
                    new_list = pickle.load(fp)
                    new_list1 = [x + (d,t) for x in new_list]
                    if t == 30:
                        def change_one(x):
                            l = list(x)
                            l[0] = 1-l[0]
                            return tuple(l)
                        new_list1 = [change_one(x) for x in new_list1]
                    new_list1 = [x + (abs(x[0]-0.5),) for x in new_list1]
                    mylist += new_list1
    sorted_list = sorted(mylist, key=lambda x: x[-1], reverse=False)  # sort in descending order (True)
    sorted_list = sorted_list[:num//2]
    df = pd.DataFrame(columns=['prob','all_files','row','col','fld','drug','drug_dose', 'AB_dose'])
    for x in sorted_list:
        df = df.append({'prob':x[0], 'all_files':x[3], 'row':x[1], 'col':x[2], 'fld':x[4], 'drug':x[5], 'drug_dose':x[6], 'AB_dose':x[7]}, ignore_index=True)
    valid_df = df
    valid_set = my_one_dataset(valid_df, transform=None, gradcam=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1)
    
    model = get_model()
    model.load_state_dict(torch.load(model_pth))
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
            
    fig = plt.figure(figsize=(20,5*num//4))
    i = 0
    for x,row,col,fname,fld,drug, drug_dose, AB_dose, label_prob in valid_loader:
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            fx = model(x)
            prob = torch.nn.Softmax(dim=1)(fx)
            y_pred = int(np.argmax(prob.cpu()))
            y_prob = float(prob[0][y_pred])
            y_pred = 'AB' if y_pred==0 else 'VEH'
        grad_cam = GradCAM(model, 'layer4');
        mask = grad_cam(x,None);
        mask = cv2.resize(mask, (2048, 2048));
        plt.subplot(num//4,4,i+1)
        plt.imshow(x.squeeze().squeeze())  #.permute(1,2,0)[:,:,:])
        plt.title('drug %s %d, AB %d' %(str(drug[0]), drug_dose, AB_dose))
        plt.subplot(num//4,4,i+2)
        plt.imshow(x.squeeze().squeeze())
        plt.imshow(mask, alpha=0.6, cmap='magma')
        plt.title('prob wrong %.2f, row %s col %s fld %s' %(label_prob, str(np.array(row)[0]), str(col[0]), str(fld[0]).replace('tensor','').replace('([','').replace('])','')))
        i += 2
    plt.show()
    
