import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import tqdm
import seaborn as sn

class yasai:
    def checkpath(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def visualize(**images):
        n = len(images)
        plt.figure(figsize=(16, 5))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()

    def round(temp):
        return np.round((temp - np.min(temp))/((np.max(temp) - np.min(temp))))

    def show_v2(dataset, idx, model=None):
        image, mask = dataset[idx]
        if model is not None:
            pred = model(image.unsqueeze(0))
            with torch.no_grad():
                pred = np.asarray(pred).squeeze()
        with torch.no_grad():
            image = np.asarray(image).transpose(1, 2, 0)
            mask = np.asarray(mask)

        if model is not None:
            tempdict = {}
            tempdict['image'] = image
            for i in range(pred.shape[0]):
                tempdict['pred_' + str(i)] = 0.4 * yasai.round(pred[i]) + 0.6 * image[...,0].squeeze()
            yasai.visualize(**tempdict)

        tempdict = {}
        tempdict['image'] = image
        for i in range(mask.shape[0]):
            tempdict['mask_' + str(i)] = 0.4 * yasai.round(mask[i]) + 0.6 * image[...,0].squeeze()
        yasai.visualize(**tempdict)

    def model_save_v1(model, text=''):
        temp = os.path.join(os.getcwd(), 'model_' + text + datetime.now().strftime("%y%m%d%H%M.pt"))
        torch.save({'state_dict': model.state_dict(), 'model': model}, temp)
        print('Successfully saved to ' + temp)

    def model_load_v1(path):
        temp = torch.load(path)
        model = temp['model']
        model.load_state_dict(temp['state_dict'])
        print('Successfully loaded from ' + path)
        return model

    def compute_iou_v1(pred, label):
        # print(label.shape, np.unique(label))
        # print(round(pred).shape, np.unique(round(pred)))
        label_c = label == 1
        pred_c = round(pred) == 1

        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()

        if union != 0 and np.sum(label_c) != 0:
            return intersection / union
        
    def compute_batch_iou_v1(model, data_loader):
        ious = []
        for image, mask in tqdm(data_loader, desc='Iterating'):
            pred = model(image)
            with torch.no_grad():
                pred = np.asarray(pred).squeeze()
                mask = np.asarray(mask)
            ious += [yasai.compute_iou_v1(pred, mask)]
        print(sum(ious)/len(ious))

    def get_path_df_v1(img_path, mask_path):
        images, masks = [], []

        i = 0

        for get_img_name in os.listdir(img_path):
            images += [os.path.join(img_path, get_img_name)] # NORMAL_G1_Lid1_LRid293_Gid3133_Bl30.png
            masks += [os.path.join(mask_path, get_img_name.replace(get_img_name.split('_')[-1], 'C4.png'))] # NORMAL_G1_Lid1_LRid293_Gid3133_C4.png
            
            i = i+1

        PathDF = pd.DataFrame({'images': images, 'masks': masks})
        PathDF = PathDF.sample(frac=1).reset_index(drop=True)
        print('got ' + str(i))
        PathDF.head()
        return PathDF

    def model_checker_v1(model, tenser, dataloader=None, device='cuda'):
        model = model.to(device)
        t = torch.randn(tenser).to(device)
        print('give_model', t.shape)
        get = model(t)
        print('got_from_model', get.shape)
        if dataloader is not None:
            for data, label in dataloader:
                print()
                print('data_shape', data.shape)
                print('label_shape', label.shape)
                break

    def success():
        print('\n\nSuccessfully Completed!!!\n\n')

    def confusion_matrix(dataset, model, CLASSES):
        confusion_matrix = np.zeros((len(CLASSES),len(CLASSES)))
        for image, label in dataset:
            pred = model(image.unsqueeze(0).cuda())
            with torch.no_grad():
                pred = np.asarray(pred.cpu()).squeeze()

            pred_index = np.argmax(pred)
            label_index = np.argmax(label)

            confusion_matrix[label_index][pred_index] += 1 #-|
        
        accuracy = 0
        for i in range(len(confusion_matrix)):
            accuracy += confusion_matrix[i][i]
        accuracy = accuracy / sum(sum(confusion_matrix))
        print('accuracy = ' + str(accuracy))

        avg_precision = 0
        for i in range(len(confusion_matrix)):
            avg_precision += confusion_matrix[i][i] / (sum(confusion_matrix)[i] + 1e-9)
        avg_precision = avg_precision / len(confusion_matrix)
        print('avg_precision = ' + str(avg_precision))

        avg_recall = 0
        for i in range(len(confusion_matrix)):
            avg_recall += confusion_matrix[i][i] / (sum(confusion_matrix[i]) + 1e-9)
        avg_recall = avg_recall / len(confusion_matrix)
        print('avg_recall = ' + str(avg_recall))

        df_confusion_matrix = pd.DataFrame(confusion_matrix, index = ['Actual_' + i for i in CLASSES], columns = ['Pred_' + i for i in CLASSES])
        plt.figure(figsize = (6, 5))
        sn.heatmap(df_confusion_matrix, annot=True, cmap=sn.cubehelix_palette(as_cmap=True))
        return confusion_matrix