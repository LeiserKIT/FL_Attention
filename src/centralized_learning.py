# ----------
# imports
# ----------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tensorboardX import SummaryWriter
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import pandas as pd
import random
import os
import datetime
import gc as garb



# ----------
# models
# ----------

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    Analogously for ResNet101 below.
    """
    def __init__(self):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained = True)
        self.densenet121.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class ResNet101(nn.Module):
    def __init__(self):
        super(ResNet101, self).__init__()
        self.resnet101 = torchvision.models.resnet101(pretrained = True)
        self.resnet101.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Sequential(
            nn.Linear(num_ftrs, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet101(x)
        return x



# ----------
# dataset
# ----------

# help function for downscaling of images
def resize_image(image, scale_percent):
    target_width = int(image.size[0] * scale_percent / 100) 
    target_height = int(image.size[1] * scale_percent / 100)
    target_dim = (target_height, target_width)
    resized_image = transforms.Resize(size=target_dim)(image)
    return resized_image

# dataset
class ChestXRays(Dataset):
    def __init__(self, csv_file, root_dir, train, scale_percent, num_samples=None):
        '''
        args:
            csv_file (string): path to csv_file with path to label mappings and image sizes
            root_dir (string): directory where images are actually stored
            train (boolean): decides which transformations are applied
            scale_percent (int): scaling factor (percentage) --> we are trying 10 first
            num_samples (int): determines how many samples we are training with, if 'None' --> full dataset
        '''
        self.df_label_mapping = pd.read_csv(csv_file)
        self.num_samples = num_samples
        if self.num_samples == None:
            self.df_label_mapping = self.df_label_mapping.loc[self.df_label_mapping['width']==2828].loc[self.df_label_mapping['height']==2320].reset_index(drop=True)
        else:
            self.df_label_mapping = self.df_label_mapping.loc[self.df_label_mapping['width']==2828].loc[self.df_label_mapping['height']==2320].reset_index(drop=True).head(self.num_samples)
        self.root_dir = root_dir
        self.train = train
        self.scale_percent = scale_percent       
        self.input_dim = int(2320 * scale_percent / 100) * 0.875

    def __len__(self):
        return len(self.df_label_mapping)

    def __getinputdim__(self):
        return self.input_dim

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.df_label_mapping.loc[index, 'path'])  
        image = Image.open(img_name)

        # transform
        if self.train == True:
            image = resize_image(image, self.scale_percent)          
            rcrop = transforms.RandomCrop(int(0.875*image.size[1]))     # 0.875=224/256 (factor used by Irvin et al. (2019)), image.size[1] is shorter side of image  --> corresponds to 203 in case of 10% scaling
            image = rcrop(image)
            tensorize = transforms.ToTensor()
            image = tensorize(image)
            MEAN = 0.5065
            STD = 0.2863
            normalize = transforms.Normalize(MEAN, STD) 
            image = normalize(image)
        elif self.train == False:
            image = resize_image(image, self.scale_percent)
            ccrop = transforms.CenterCrop(int(0.875*image.size[1]))
            image = ccrop(image)
            tensorize = transforms.ToTensor()
            image = tensorize(image)
            normalize = transforms.Normalize(0.5065, 0.2894)
            image = normalize(image)

        # labels
        labels = torch.FloatTensor()
        if self.df_label_mapping.loc[index, 'Cardiomegaly'] == 1:
            labels = torch.cat((labels,torch.FloatTensor([1])))
        else:
            labels = torch.cat((labels,torch.FloatTensor([0])))
        if self.df_label_mapping.loc[index, 'Pleural_Effusion'] == 1:
            labels = torch.cat((labels,torch.FloatTensor([1])))
        else:
            labels = torch.cat((labels,torch.FloatTensor([0])))
        if self.df_label_mapping.loc[index, 'Edema'] == 1:
            labels = torch.cat((labels,torch.FloatTensor([1])))
        else:
            labels = torch.cat((labels,torch.FloatTensor([0])))
        if self.df_label_mapping.loc[index, 'Atelectasis'] == 1:
            labels = torch.cat((labels,torch.FloatTensor([1])))
        else:
            labels = torch.cat((labels,torch.FloatTensor([0])))
        if self.df_label_mapping.loc[index, 'Consolidation'] == 1:
            labels = torch.cat((labels,torch.FloatTensor([1])))
        else:
            labels = torch.cat((labels,torch.FloatTensor([0])))
        sample = (image, labels)
        return sample


def load_data(sim, sc, num_samples, bs):
    # base
    if sim == -1:
        train_dataset = ChestXRays(csv_file="../data/train/df_train.csv", root_dir="../data/CheXpert/", train=True, scale_percent=sc, num_samples=num_samples)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_)
        valid_dataset = ChestXRays(csv_file="../data/train/df_valid.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_dataset = ChestXRays(csv_file="../data/test/df_test.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # sim0
    if sim == 0:
        train_dataset = ChestXRays(csv_file="../data/train/FL/sim0/train/df_train_cl.csv", root_dir="../data/CheXpert/", train=True, scale_percent=sc, num_samples=num_samples)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_)
        valid_dataset = ChestXRays(csv_file="../data/train/FL/sim0/valid/df_valid_cl.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_dataset = ChestXRays(csv_file="../data/test/df_test.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # sim1
    if sim == 1:
        train_dataset = ChestXRays(csv_file="../data/train/FL/sim1/train/df_train_cl.csv", root_dir="../data/CheXpert/", train=True, scale_percent=sc, num_samples=num_samples)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_)
        valid_dataset = ChestXRays(csv_file="../data/train/FL/sim1/valid/df_valid_cl.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_dataset = ChestXRays(csv_file="../data/test/df_test.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # sim2
    if sim == 2:
        train_dataset = ChestXRays(csv_file="../data/train/FL/sim2/train/df_train_cl.csv", root_dir="../data/CheXpert/", train=True, scale_percent=sc, num_samples=num_samples)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_)
        valid_dataset = ChestXRays(csv_file="../data/train/FL/sim2/valid/df_valid_cl.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_dataset = ChestXRays(csv_file="../data/test/df_test.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)

    # sim3
    if sim == 3:
        train_dataset = ChestXRays(csv_file="../data/train/FL/sim3/train/df_train_cl.csv", root_dir="../data/CheXpert/", train=True, scale_percent=sc, num_samples=num_samples)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_)
        valid_dataset = ChestXRays(csv_file="../data/train/FL/sim3/valid/df_valid_cl.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=8)
        test_dataset = ChestXRays(csv_file="../data/test/df_test.csv", root_dir="../data/CheXpert/", train=False, scale_percent=sc, num_samples=num_samples)
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=8)
    
    return train_dataset, test_dataset, trainloader, validloader, testloader



# ----------
# helper functions
# ----------

def compute_metrics(pred, labels, losses, test=False):
    n_classes = pred.shape[1]
    fpr, tpr, aucs, precision, recall, accuracy = {}, {}, {}, {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels[:,i], pred[:,i])
        aucs[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(labels[:,i], pred[:,i])
        fpr[i], tpr[i], precision[i], recall[i] = fpr[i].tolist(), tpr[i].tolist(), precision[i].tolist(), recall[i].tolist()
    if test:
        for j in np.around(np.arange(0.05, 1, 0.05),2).tolist():
            correct = (np.array(pred > j, dtype=float) == np.array(labels)).sum()
            total = len(labels)*5
            accuracy[j] = correct / total
    else:
        correct = (np.array(pred > 0.4, dtype=float) == np.array(labels)).sum()
        total = len(labels)*5
        accuracy = correct / total

    metrics = {'fpr': fpr,
               'tpr': tpr,
               'aucs': aucs,
               'precision': precision,
               'recall': recall,
               'loss': dict(enumerate(losses.tolist())),
               'accuracy': accuracy}

    return metrics


def diff_per_pixel(a1: np.array, a2: np.array):
    dim = a1.shape[0]
    return sum(sum(np.abs(a1-a2)))/(dim*dim)

    
def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)



# ----------
# main
# ----------
def main(mod, sim, num_samples, epochs, bs=16, val_interv=200, gc=True):
    # time
    start_time = datetime.datetime.now()

    ### PARAMS
    MODEL = mod
    SIM = sim
    NUM_SAMPLES = num_samples        # None for full dataset
    EPOCHS = epochs
    BATCH_SIZE = bs
    LR = 0.0001
    SCALE_PERCENT = 10
    VALIDATION = True
    VAL_INTERV = val_interv
    GRADCAM = gc

    assert MODEL in ['DenseNet121', 'ResNet101'], 'Invalid model. Choose between DenseNet121 and ResNet101.'
    if NUM_SAMPLES is not None:
        log_num_samples = NUM_SAMPLES
    else:
        log_num_samples = 'all'

    OUTPUT_DIR = 'logs/{}_SIM{}_{}_E{}_BS{}_SC{}_'.format(MODEL[0], SIM, log_num_samples, EPOCHS, BATCH_SIZE, SCALE_PERCENT) + datetime.datetime.now().strftime("%d.%m.%y-%H:%M:%S") 
    print(OUTPUT_DIR)


    ### DATASET
    # num_workers = 4*numGPUs
    print('load data...')
    train_dataset, test_dataset, trainloader, validloader, testloader = load_data(sim=SIM, sc=SCALE_PERCENT, num_samples=NUM_SAMPLES, bs=BATCH_SIZE)
    print('done\n')


    ### MODEL
    print('load model...')
    if MODEL == 'DenseNet121':
        model = DenseNet121()
    elif MODEL == 'ResNet101':
        model = ResNet101()
    optimizer = optim.Adam(model.parameters(), lr = LR, betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
    criterion = torch.nn.BCELoss()
    print('done\n')


    ### GPU
    print('attempting to move to GPU...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ' + str(device))
    device_ids = [0,1]
    model = nn.DataParallel(model, device_ids=device_ids).to(device)
    criterion.to(device)
    print('done\n')


    ### LOGGING
    writer = SummaryWriter(logdir=OUTPUT_DIR)
    summary_file = open(OUTPUT_DIR+"/summary.txt", "w+") 
    writer.add_text('Model', MODEL)
    summary_file.write("Model: {}\n".format(MODEL))
    writer.add_text('Batch Size', str(BATCH_SIZE))
    summary_file.write("Batch Size: {}\n".format(BATCH_SIZE))
    writer.add_text('Epochs', str(EPOCHS))
    summary_file.write("Epochs: {}\n".format(EPOCHS))
    writer.add_text('Learning Rate', str(LR))
    summary_file.write("Learning Rate: {}\n".format(LR))
    writer.add_text('Scaling Percentage', str(SCALE_PERCENT))
    summary_file.write("Scaling Percentage: {}\n\n".format(SCALE_PERCENT))
    writer.add_text('Num Samples', str(NUM_SAMPLES))
    summary_file.write("Num Samples: {}\n\n".format(NUM_SAMPLES))
    writer.add_text('Input dim', str(int(train_dataset.input_dim))+'x'+str(int(train_dataset.input_dim)))
    summary_file.write("Input dim: {}x{}\n\n".format(int(train_dataset.input_dim), int(train_dataset.input_dim)))


    ### TRAINING
    print('start training...')
    train_step = 0    # +1 after every training batch
    val_step = 0      # +1 after every full validation

    for epoch in range(EPOCHS):
        
        model.train()
        preds_list, labels_list, losses_list = [], [], []

        for i, data in enumerate(trainloader):
            print('Epoch {}/{}: Training - Batch {}/{}.                                      '.format(epoch+1, EPOCHS, i+1, len(trainloader)), end='\r')

            # warm up
            optimizer.zero_grad()
            input, labels = data
            input, labels = input.to(device), labels.to(device)

            # forward
            pred = model(input)
            loss = criterion(pred, labels)
            writer.add_scalar('train/loss', loss.item(), train_step) 

            preds_list += [pred.cpu()]
            labels_list += [labels.cpu()]
            losses_list += [loss.cpu()] 

            # backward
            loss.backward()
            optimizer.step()

            train_step += 1

            ### VALIDATION
            if VALIDATION:
                if (i+1) % VAL_INTERV == 0 or i+1 == len(trainloader):
                    
                    model.eval()
                    with torch.no_grad():
                        preds_val_list, labels_val_list, losses_val_list = [], [], []

                        for j, data in enumerate(validloader, 0):
                            print('Epoch {}/{}: Training - Batch {}/{}. Validation - Batch {}/{}.'.format(epoch+1, EPOCHS, i+1, len(trainloader), j+1, len(validloader)), end='\r')
                            
                            input_val, labels_val = data
                            input_val, labels_val = input_val.to(device), labels_val.to(device)
                            pred_val = model(input_val)

                            # validation loss
                            loss_val = criterion(pred_val, labels_val) 

                            preds_val_list += [pred_val.cpu()]
                            labels_val_list += [labels_val.cpu()]   
                            losses_val_list += [loss_val.cpu()]       

                        # val eval_metrics
                        preds_val_list, labels_val_list, losses_val_list = torch.cat(preds_val_list), torch.cat(labels_val_list), torch.stack(losses_val_list)
                        eval_metrics_val = compute_metrics(preds_val_list, labels_val_list, losses_val_list)  

                        # log loss, acc, auc
                        writer.add_scalar('val/loss', np.mean(list(eval_metrics_val['loss'].values())), val_step)
                        writer.add_scalar('val/acc', eval_metrics_val['accuracy'], val_step)
                        for k, v in eval_metrics_val['aucs'].items():
                            writer.add_scalar('val/auc_class_{}'.format(k), v, val_step)
                        
                        val_step += 1

                        # switch back to train mode
                        model.train()
                        # end of validation

        # train eval_metrics 
        preds_list, labels_list, losses_list = torch.cat(preds_list), torch.cat(labels_list), torch.stack(losses_list)
        eval_metrics = compute_metrics(preds_list.detach(), labels_list, losses_list)

        # log epoch accuracy & loss
        writer.add_scalar('train/acc', eval_metrics['accuracy'], epoch)
        writer.add_scalar('train/epoch loss', np.mean(losses_list.tolist()), epoch)

        # end of epoch
        print('')
        print('Epoch {}/{}: finished'.format(epoch+1, EPOCHS))

    print('done\n')


    ### TESTING
    print('start testing...')
    summary_file.write('TEST RESULTS\n')

    model.eval() 
    with torch.no_grad():
        inputs_test_list, preds_test_list, labels_test_list, losses_test_list = [], [], [], []

        for k, data in enumerate(testloader, 0):
            print('Testing - Batch {}/{}.'.format(k+1, len(testloader)), end='\r')

            input_test, labels_test = data
            input_test, labels_test = input_test.to(device), labels_test.to(device)
            pred_test = model(input_test)

            # testing loss
            loss_test = criterion(pred_test, labels_test)

            inputs_test_list += [input_test.cpu()]
            preds_test_list += [pred_test.cpu()]
            labels_test_list += [labels_test.cpu()]   
            losses_test_list += [loss_test.cpu()]

        # test eval metrics
        inputs_test_list, preds_test_list, labels_test_list, losses_test_list = torch.cat(inputs_test_list), torch.cat(preds_test_list), torch.cat(labels_test_list), torch.stack(losses_test_list)
        eval_metrics_test = compute_metrics(preds_test_list, labels_test_list, losses_test_list, test=True)
        # accuracy histogram
        fig = plt.figure(figsize=(10,6))
        plt.bar(range(len(list(eval_metrics_test['accuracy'].keys()))), list(eval_metrics_test['accuracy'].values()))
        plt.title('Accuracy on thresholds')
        plt.xlabel('Thresholds')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(list(eval_metrics_test['accuracy'].keys()))), list(eval_metrics_test['accuracy'].keys()))

        # log metrics
        writer.add_text('test/accuracy', str(eval_metrics_test['accuracy'])) 
        writer.add_text('test/loss', str(np.mean(list(eval_metrics_test['loss'].values()))))  
        writer.add_text('test/AUC', str(eval_metrics_test['aucs']))
        writer.add_text('test/precision', str(eval_metrics_test['precision']))
        writer.add_text('test/recall', str(eval_metrics_test['recall']))
        summary_file.write('test/accuracy: {}\n'.format(eval_metrics_test['accuracy']))
        summary_file.write('test/eval_metrics: {}'.format(eval_metrics_test))
        writer.add_figure('Test Accuracy Thresholds', fig)
        plt.clf()
        plt.close(fig)
        garb.collect()

        # ROC
        fig, axs = plt.subplots(2, len(labels_test_list[0]), figsize=(24,12))
        for i, (fpr, tpr, aucs, precision, recall, label) in enumerate(zip(eval_metrics_test['fpr'].values(), eval_metrics_test['tpr'].values(),
                                                                            eval_metrics_test['aucs'].values(), eval_metrics_test['precision'].values(),
                                                                            eval_metrics_test['recall'].values(), ['Cardiomegaly', 'Pleural Effusion', 'Edema', 'Atelectasis', 'Consolidation'])):
            # top row -- ROC
            axs[0,i].plot(fpr, tpr, label='AUC = %0.2f' % aucs)
            axs[0,i].plot([0, 1], [0, 1], 'k--')  # diagonal margin
            axs[0,i].set_xlabel('False Positive Rate')
            # bottom row - Precision-Recall
            axs[1,i].step(recall, precision, where='post')
            axs[1,i].set_xlabel('Recall')
            # format
            axs[0,i].set_title(label)
            axs[0,i].legend(loc="lower right")   
        plt.suptitle('roc_pr')
        axs[0,0].set_ylabel('True Positive Rate')
        axs[1,0].set_ylabel('Precision')
        for ax in axs.flatten():
            ax.set_xlim([0.0, 1.05])
            ax.set_ylim([0.0, 1.05])
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR + '/roc_pr.png', pad_inches=0.)
        writer.add_figure('ROC_PR', fig)
        plt.clf()
        plt.close(fig)
        garb.collect()

        print('')
        print('done\n')


    # GradCAM & difference metrics
    if GRADCAM:
        print('start GradCAM...')
        if MODEL == 'ResNet101':
            target_layer = [model.module.resnet101.layer4[2].conv3]
        if MODEL == 'DenseNet121':
            target_layer = [model.module.densenet121.features.denseblock4.denselayer16.conv2] 
        gradcam = GradCAM(model=model, target_layers=target_layer)
        log_list = []

        # make directory for figures
        save_path = './{}/figures'.format(OUTPUT_DIR)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # paths in test data
        paths = [path for path in test_dataset.df_label_mapping['path'].tolist()]
        label_mapping = {0: 'Cardiomegaly', 1: 'Pleural Effusion', 2: 'Edema', 3: 'Atelectasis', 4: 'Consolidation'}

        # open df with box coordinates
        df_boxes = pd.read_pickle('../data/test/df_boxes.pkl')
        df_boxes = df_boxes[df_boxes['path'].isin(paths)]

        # iterate over every image in test data
        for image_idx in range(len(inputs_test_list)):
            print('GradCAM - Image {}/{}.'.format(image_idx+1, len(inputs_test_list)), end='\r')
            image = inputs_test_list[image_idx].unsqueeze(dim=0)    # one image
            labels = labels_test_list[image_idx]                    # five labels
            preds = preds_test_list[image_idx]
            path = paths[image_idx]                                 # one path

            # iterate over five labels of one image and only act if label is one --> diagnosis exists, therefore box coordinates also exist
            for label_idx in range(len(labels)): 
                if labels[label_idx] == 1:
                    target = [ClassifierOutputTarget(label_idx)]
                    grayscale_cam = gradcam(input_tensor=image, targets=target)[0]

                    log_list_item = []
                    log_list_item.append(path)
                    log_list_item.append(label_mapping[label_idx])
                    log_list_item.append(preds[label_idx].item())

                    ### visualization in color
                    # create heatmap from gradcam
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    heatmap = cv2.resize(heatmap, (2320,2320))

                    # counterheatmap for visualization: start as full blue, add red within box coordinates 
                    counterheatmap_vis = np.full(shape=(2320, 2828, 3), fill_value=[0, 0, 0.5647], dtype='float32')
                    counterheatmap_vis = cv2.applyColorMap(np.uint8(counterheatmap_vis), cv2.COLORMAP_JET)              # JET has 64 colors: dark blue (0,0,144) and dark red (128,0,0) at edges
                    counterheatmap_vis = cv2.cvtColor(counterheatmap_vis, cv2.COLOR_BGR2RGB)

                    # analogous for counterheatmaps for comparison: start as full 0, add 1 within box coordinates
                    counterheatmap_comp = np.full(shape=(2320,2828), fill_value=0, dtype='float32')
                    
                    # iterate over boxes for one label (could be more than one box per label)
                    boxes = df_boxes[df_boxes['path']==path].iloc[:,1:].values[0][label_idx]
                    for box in boxes:
                        xmin, ymin, xmax, ymax = [box[val] for val in (0,1,2,3)]
                        counterheatmap_vis[ymin:ymax+1,xmin:xmax+1] = [128,0,0]
                        counterheatmap_comp[ymin:ymax+1,xmin:xmax+1] = 1
                    counterheatmap_vis = np.float32(counterheatmap_vis) / 255
                    # imitate centercrop: cut 508 columns --> [254,2574]
                    counterheatmap_vis = counterheatmap_vis[:,254:2574]
                    counterheatmap_comp = counterheatmap_comp[:,254:2574]  

                    # plot: original image, heatmap, combination, counterheatmap
                    fig = plt.figure(figsize=(24,7))
                    fig.suptitle(path[6:-4] + '\nLabel: {} (pred={:04.2f})'.format(label_mapping[label_idx], preds[label_idx]), fontsize=25, fontweight='bold')
                    rows, columns = 1, 5
                    original = Image.open('../data/CheXpert/' + path)
                    original = (np.float32(asarray(original.convert('RGB'))) / 255)[:,254:2574]
                    fig.add_subplot(rows,columns,1)
                    plt.imshow(original)
                    plt.axis('off')
                    plt.title('Original')
                    fig.add_subplot(rows,columns,2)
                    plt.imshow(0.5 * original + 0.5 * heatmap)
                    plt.axis('off')
                    plt.title('Original with GradCAM')
                    fig.add_subplot(rows,columns,3)
                    plt.imshow(heatmap)
                    plt.axis('off')
                    plt.title('GradCAM')
                    fig.add_subplot(rows,columns,5)
                    plt.imshow(counterheatmap_vis)
                    plt.axis('off')
                    plt.title('Expert Annotation')

                    ### difference metrics
                    # Diff 1: Average difference per pixel
                    grayscale_cam_comp = cv2.resize(grayscale_cam, dsize=(2320,2320))
                    diff = diff_per_pixel(grayscale_cam_comp, counterheatmap_comp)
                    log_list_item.append(diff)
                    fig.text(0.148, 0.9, 'Difference per pixel = {:05.3f}'.format(diff), fontsize=15, bbox={'facecolor': 'white', 'pad': 10})
                    # Diff 2: Intersection percentage - percentage of pixels within expert box that are also 1 in (transformed) grayscale cam || Intersection score - intersection percentage * percentage of total grayscale cam 1s that are within expert box
                    # prepare transformed gradcam
                    grayscale_cam_comp[grayscale_cam_comp > 0.5] = 1    # threshold 0.5
                    grayscale_cam_comp[grayscale_cam_comp != 1] = 0
                    grayscale_cam_comp_resized = np.pad(grayscale_cam_comp, (254,254), mode='constant')[254:-254,:]
                    # plot transformed heatmap
                    heatmap_transformed = cv2.applyColorMap(np.uint8(255 * grayscale_cam_comp), cv2.COLORMAP_JET)
                    heatmap_transformed = np.float32(heatmap_transformed) / 255
                    heatmap_transformed = cv2.cvtColor(heatmap_transformed, cv2.COLOR_BGR2RGB)
                    fig.add_subplot(rows,columns,4)
                    plt.imshow(heatmap_transformed)
                    plt.axis('off')
                    plt.title('GradCAM (transformed)')
                    # box_1s (pixels in box) and cam_box_1s (pixels that are 1 in heatmap and inside the box coordinates of expert labeler)
                    cam_box_1s = 0
                    box_1s = 0
                    for box in boxes:
                        xmin, ymin, xmax, ymax = [box[val] for val in (0,1,2,3)]
                        unique, counts = np.unique(grayscale_cam_comp_resized[ymin:ymax+1,xmin:xmax+1], return_counts=True)
                        if len(counts) == 2:
                            cam_box_1s += counts[1]
                            box_1s += counts[0] + counts[1]
                        else:
                            box_1s += counts[0]
                            if unique[0] == 1:
                                cam_box_1s += counts[0]
                    # cam_tot_1s: total amount of activated pixels in heatmap
                    _, counts = np.unique(grayscale_cam_comp, return_counts=True)
                    if len(counts) == 2:
                        cam_tot_1s = counts[1]
                    else:
                        cam_tot_1s = 0
                    # scores
                    try:
                        intersection_perc = cam_box_1s/box_1s
                    except ZeroDivisionError:
                        intersection_perc = 0
                    try:
                        intersection_score = intersection_perc * (cam_box_1s / cam_tot_1s)
                    except ZeroDivisionError:
                        intersection_score = 0
                    log_list_item.append(intersection_perc)
                    log_list_item.append(intersection_score)
                    log_list.append(log_list_item)
                    fig.text(0.76, 0.88, '     Intersection: {:03.1f}%\nIntersection score = {:05.3f}'.format(intersection_perc*100, intersection_score), fontsize=15, bbox={'facecolor': 'white', 'pad': 10})
                    
                    # save figure
                    split_path = path[:-4].split('/')[1:]
                    fig.savefig(save_path + '/{}-{}-{}-L{}.jpg'.format(split_path[0], split_path[1], split_path[2], label_idx))
                    plt.clf()
                    plt.close(fig)
                    garb.collect()

        ### aggregated difference metrics
        df_diffs = pd.DataFrame(data=log_list, columns=['path', 'Label', 'pred', 'DpP', 'Intersection', 'Intersection Score'])
        df_diffs.to_csv(OUTPUT_DIR + '/diff_metrics.csv', index=False)
        # average DpP, Int, IntS
        dpp_mean = df_diffs['DpP'].mean()
        intp_mean = df_diffs['Intersection'].mean()
        ints_mean = df_diffs['Intersection Score'].mean()
        means = [dpp_mean, intp_mean, ints_mean]
        # average DpP, Int, IntS for pred larger and smaller 0.4
        df_diffs_conf = df_diffs[df_diffs['pred'] > 0.4]
        dpp_mean_conf = df_diffs_conf['DpP'].mean()
        intp_mean_conf = df_diffs_conf['Intersection'].mean()
        ints_mean_conf = df_diffs_conf['Intersection Score'].mean()
        means_conf = [dpp_mean_conf, intp_mean_conf, ints_mean_conf]
        df_diffs_shy = df_diffs[df_diffs['pred'] < 0.4]
        dpp_mean_shy = df_diffs_shy['DpP'].mean()
        intp_mean_shy = df_diffs_shy['Intersection'].mean()
        ints_mean_shy = df_diffs_shy['Intersection Score'].mean()
        means_shy = [dpp_mean_shy, intp_mean_shy, ints_mean_shy]
        # average DpP, Int, IntS for each Label
        df_diffs_ca = df_diffs[df_diffs['Label'] == 'Cardiomegaly']
        dpp_mean_ca = df_diffs_ca['DpP'].mean()
        intp_mean_ca = df_diffs_ca['Intersection'].mean()
        ints_mean_ca = df_diffs_ca['Intersection Score'].mean()
        means_ca = [dpp_mean_ca, intp_mean_ca, ints_mean_ca]
        df_diffs_p = df_diffs[df_diffs['Label'] == 'Pleural Effusion']
        dpp_mean_p = df_diffs_p['DpP'].mean()
        intp_mean_p = df_diffs_p['Intersection'].mean()
        ints_mean_p = df_diffs_p['Intersection Score'].mean()
        means_p = [dpp_mean_p, intp_mean_p, ints_mean_p]
        df_diffs_e = df_diffs[df_diffs['Label'] == 'Edema']
        dpp_mean_e = df_diffs_e['DpP'].mean()
        intp_mean_e = df_diffs_e['Intersection'].mean()
        ints_mean_e = df_diffs_e['Intersection Score'].mean()
        means_e = [dpp_mean_e, intp_mean_e, ints_mean_e]
        df_diffs_a = df_diffs[df_diffs['Label'] == 'Atelectasis']
        dpp_mean_a = df_diffs_a['DpP'].mean()
        intp_mean_a = df_diffs_a['Intersection'].mean()
        ints_mean_a = df_diffs_a['Intersection Score'].mean()
        means_a = [dpp_mean_a, intp_mean_a, ints_mean_a]
        df_diffs_co = df_diffs[df_diffs['Label'] == 'Consolidation']
        dpp_mean_co = df_diffs_co['DpP'].mean()
        intp_mean_co = df_diffs_co['Intersection'].mean()
        ints_mean_co = df_diffs_co['Intersection Score'].mean()
        means_co = [dpp_mean_co, intp_mean_co, ints_mean_co]
        # log
        summary_file.write("Means of difference metrics [Difference per Pixel, Intersection Percentage, Intersection Score]\n")
        summary_file.write("Total: {}\n".format(means))
        writer.add_text('gradcam/means total', str(means))
        summary_file.write("pred > 0.5: {}\n".format(means_conf))
        writer.add_text('gradcam/means (pred > 0.4)', str(means_conf))
        summary_file.write("pred < 0.5: {}\n".format(means_shy))
        writer.add_text('gradcam/means (pred < 0.4)', str(means_shy))
        summary_file.write("Cardiomegaly: {}\n".format(means_ca))
        writer.add_text('gradcam/ means Cardiomegaly', str(means_ca))
        summary_file.write("Pleural Effusion: {}\n".format(means_p))
        writer.add_text('gradcam/ means Pleural Effusion', str(means_p))
        summary_file.write("Edema: {}\n".format(means_e))
        writer.add_text('gradcam/ means Edema', str(means_e))
        summary_file.write("Atelectasis: {}\n".format(means_a))
        writer.add_text('gradcam/ means Atelectasis', str(means_a))
        summary_file.write("Consolidation: {}\n".format(means_co))
        writer.add_text('gradcam/ means Consolidation', str(means_co))

    # end of testing


    # time
    end_time = datetime.datetime.now()
    delta = end_time - start_time
    s = delta.seconds
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_elapsed = '{:02}:{:02}:{:02}h'.format(int(hours), int(minutes), int(seconds))
    writer.add_text('Runtime', time_elapsed) 

    ### END OF RUN
    writer.close()
    summary_file.close()

    print('\ndone in {}'.format(time_elapsed))



# mod: 'DenseNet121', 'ResNet101'
# num_samples = None for all data
if __name__ == "__main__":
    main(mod='ResNet101', sim=3, num_samples=None, epochs=10, bs=16, val_interv=200, gc=True)