import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import glob
import cv2
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.colors

from medpy.metric.binary import hd, dc

from scipy import stats

import statsmodels.api as sm

import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc

import scikitplot as skplt



####################################################################################################################################################################

#np.set_printoptions(threshold=sys.maxsize)

seed = 4558
np.random.seed = seed

height=384
width=384
channels=1

axis =3

"""
axis here tells the class all of the code below works on !
axis = 0   background
axis = 1   right lung
axis = 2   left lung
axis = 3   covid-19 affected areas/ diseased areas

"""

####################################################################################################################################################################
# importing the coustom metric we used in our model
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

####################################################################################################################################################################


#model = tf.keras.models.load_model("U-Net COVID19 Segment model")
model = tf.keras.models.load_model("model", custom_objects={'dice_coef':dice_coef})  ## loading model
prediction = np.zeros((height,width,channels),dtype=np.float32)

img='coronacases_002.ni_z105img.png'
maskimg='coronacases_002.ni_z105msk.png'


prediction= cv2.imread(img,0)
predictionmask=cv2.imread(maskimg,0)
#print(np.unique(,return_counts=True,return_index=True))
predictionmask = (predictionmask == axis).astype(np.bool)
#print('pred check',predictionmask.max())
#imshow(predictionmask*255)
#plt.show()
#predictionmask=(predictionmask).astype(np.bool)   #######



norm_image = cv2.normalize(prediction, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  ## normalising prediction image
norm_image  = np.expand_dims(norm_image,axis=-1)
norm_image  = np.expand_dims(norm_image,axis=0)
#print(norm_image.shape,norm_image.dtype)


preds_train= model.predict(norm_image, verbose=1)
preds_train255=preds_train*255
# print(preds_train.shape,preds_train.dtype,preds_train.max(),preds_train.min())
#print(np.unique(preds_train[...,axis],return_counts=True,return_index=True))

#print(np.unique(preds_train[...,axis],return_counts=True,return_index=True))
#preds_train = (preds_train >= 230).astype(np.uint8)
#print(np.unique(preds_train[...,axis],return_counts=True,return_index=True))

#hello = (hello>145).astype(np.bool)

####################################################################################################################################################################
## this function cycles pixel intensity to get the best dice coefficient
def getbest_dice(preds_train_func,pred_mask):
	dice=np.zeros(256,dtype=np.float32)
	for i in range(0,255):
		hello=preds_train_func[...,axis].squeeze()
		hello = (hello>i).astype(np.bool)
		#ihere+=i
		dcval= dc(hello,pred_mask)*100
		#print('here',dcval)
		#dice= []
		dice[i]=dcval
		#dice= int(dice)
		#data= [dice,i]
	return dice
		#if i==255:

####################################################################################################################################################################
# Calculating Dice coef and Hauf distance

best_dice= getbest_dice(preds_train255,predictionmask)
#print(best_dice)
maxdice= max(best_dice)
print('dice coefficient',maxdice)    ############################# Val needs to be shown on GUI

#itemindex = np.where(dice==maxdata)
itemindex = np.argmax(best_dice)

#print('Intensity value at max dice',itemindex)

preds_perfect=(preds_train255>itemindex-1).astype(np.bool)
preds_perfect = preds_perfect[...,axis].squeeze()


haufdist= hd(preds_perfect,predictionmask,voxelspacing=None, connectivity=1)
print('Hausdorff Distance.',haufdist)   ############################# Val needs to be shown on GUI

####################################################################################################################################################################
# P-value and A-bland altman plot

pearson_stats=stats.pearsonr(preds_perfect.flatten(),predictionmask.flatten())
print('pearson values ','r-value :' ,pearson_stats[0],'p-value :',pearson_stats[1])     ############################# both Vals needs to be shown on GUI



mask_graph=predictionmask*255
#mask_graph=cv2.imread(maskimg,0)
#mask_graph=tf.keras.utils.to_categorical(mask_graph, num_classes=4, dtype='uint8')*255
#mask_graph= mask_graph[...,axis].squeeze()

pred_graph=(preds_train255>itemindex-1).astype(np.uint8)
pred_graph=pred_graph[...,axis].squeeze()

f, ax = plt.subplots(1, figsize = (8,5))
sm.graphics.mean_diff_plot(mask_graph.flatten(),pred_graph.flatten(), ax = ax)
plt.show()              ############################# Plot needs to be shown on GUI



####################################################################################################################################################################
# ROC/AUC curve

y_mask = cv2.imread(maskimg,0)
y_mask = tf.keras.utils.to_categorical(y_mask, num_classes=4, dtype='bool')

y_covid = y_mask[...,axis].squeeze()

#y_probas = np.squeeze(preds_train)
#y_probas = y_probas[...,axis]
y_predicted = preds_perfect

ground_truth_labels = y_covid.ravel() # we want to make them into vectors
score_value= y_predicted.ravel()

fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
roc_auc = auc(fpr,tpr)
fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive rate')
ax.set_ylabel('True Positive rate ')
ax.set_title('Receiver operating characteristic for Diseased Areas pixel wise')
ax.legend(loc="lower right")
plt.show()                         ############################# Plot needs to be shown on GUI

####################################################################################################################################################################
# Showing images of the mask and predicted mask

imshow(predictionmask)  ## original loaded mask              ############################# Plot needs to be shown on GUI
plt.show()

imshow(preds_perfect) ## predicted mask from model           ############################# Plot needs to be shown on GUI
plt.show()

heatmask=preds_train255[...,axis].squeeze()    # just a heat map to show prediction in a greater detail    ############################# Plot needs to be shown on GUI
imshow(heatmask)
plt.show()

####################################################################################################################################################################


