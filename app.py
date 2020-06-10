from tkinter import *
import tkinter.filedialog
from tkinter.ttk import Style
import tensorflow as tf
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import glob
import cv2
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from PIL import Image, ImageTk
import matplotlib.colors
from medpy.metric.binary import hd, dc
from scipy import stats
import statsmodels.api as sm
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
import scikitplot as skplt
import nibabel as nib


class App:

  def __init__(self, master):

    # All the files in selected path or selected by dialog
    # self.selected_files = []
    # self.processed_images = {}
    # self.processed_masks = {}
    # self.active_patient_id = 'patient_043'
    # self.slice_name = 4
    # self.patient_info = None
    # self.config_file = None
    # self.ed_index = '01'
    # self.es_index = ''
    # self.data_path = "./dataset/data"
    # self.images_path = "./dataset/train"
    self.model = None
    self.axis = 3
    self.total_frames = 0
    self.slice_index = 105
    # GUI related
    self.master = master
    self.create_ui()
    self.create_sidebar()
    self.load_model()
    self.init()

  def create_ui(self):
    self.master.geometry("1020x900")
    self.master.title("MRI Cardiac Segmentation")
    self.master.style = Style()
    self.master.style.theme_use("default")

    self.master.columnconfigure(0, weight=4)
    self.master.columnconfigure(1, weight=1)

    self.leftFrame = Frame(self.master, padx=10, pady=10)
    self.leftFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.rightFrame = Frame(self.master)
    self.rightFrame.columnconfigure(0, weight=1)
    self.rightFrame.grid(row=0, column=1, sticky=N+S+E+W)

    self.menuFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1, padx=10, pady=10)
    self.menuFrame.columnconfigure(0, weight=1)
    self.menuFrame.grid(row=0, column=0, sticky=N+S+E+W)

    self.metricsFrame = Frame(self.rightFrame, relief=RAISED, borderwidth=1)
    self.metricsFrame.grid(row=1, column=0, sticky=N+S+E+W)

    self.imageFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageFrame.grid(row=0, column=0)

    self.imageControlsFrame = Frame(self.leftFrame, relief=RAISED, borderwidth=1, padx=5, pady=5)
    self.imageControlsFrame.grid(row=1, column=0)

    srcLbl = Label(self.imageFrame, text="Original Mask")
    srcLbl.grid(row=0, column=0)

    self.org_mask_cmp = Label(self.imageFrame)
    self.org_mask_cmp.grid(row=1, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Predicted Mask")
    srcLbl.grid(row=2, column=0)

    self.pred_mask_cmp = Label(self.imageFrame)
    self.pred_mask_cmp.grid(row=3, column=0, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Analysis Plot")
    srcLbl.grid(row=0, column=1)

    self.analysis_cmp = Label(self.imageFrame)
    self.analysis_cmp.grid(row=1, column=1, padx=5, pady=5)

    srcLbl = Label(self.imageFrame, text="Heatmask")
    srcLbl.grid(row=2, column=1)

    self.heatmap_cmp = Label(self.imageFrame)
    self.heatmap_cmp.grid(row=3, column=1, padx=5, pady=5)

    # self.esimg = Label(self.imageFrame)
    # self.esimg.grid(row=3, column=0, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ED Predicted Mask")
    # srcLbl.grid(row=0, column=2)

    # self.edpmask = Label(self.imageFrame)
    # self.edpmask.grid(row=1, column=2, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ES Predicted Mask")
    # srcLbl.grid(row=2, column=2)

    # self.espmask = Label(self.imageFrame)
    # self.espmask.grid(row=3, column=2, padx=5, pady=5)

    b = Button(self.imageControlsFrame, text=">", command=self.next_frame)
    b.grid(row=1, column=1, sticky=E)
    b = Button(self.imageControlsFrame, text="<", command=self.prev_frame)
    b.grid(row=1, column=0, sticky=W)

  def create_sidebar(self):
    fselect_button = Button(self.menuFrame, text="Load Data", command=self.open_path_dialog)
    fselect_button.grid(row=0, sticky=W+E)

    self.exit_button = Button(
      master=self.menuFrame,
      text="Exit",
      command=self.master.quit
    )

    self.exit_button.grid(row=1, sticky=W+E)

    # self.patientNameLbl = Label(self.metricsFrame, text="", anchor="center")
    # self.patientNameLbl.grid(row=5, column=0, padx=10, pady=10)

    # tbl_fspace = Label(self.metricsFrame, text="")
    # tbl_fspace.grid(row=0, column=0, padx=5, pady=5)

    # lv_ed_lbl = Label(self.metricsFrame, text="LV")
    # lv_ed_lbl.grid(row=1, column=0, padx=5, pady=5)

    # dice_lbl = Label(self.metricsFrame, text="Dice coeff.")
    # dice_lbl.grid(row=2, column=0, padx=5, pady=5)

    # volume_lbl = Label(self.metricsFrame, text="Volume")
    # volume_lbl.grid(row=3, column=0, padx=5, pady=5)

    # volume_lbl = Label(self.metricsFrame, text="EF")
    # volume_lbl.grid(row=4, column=0, padx=5, pady=5)

    # ed_tbl_lbl = Label(self.metricsFrame, text="ED")
    # ed_tbl_lbl.grid(row=0, column=1, padx=5, pady=5)
    # es_tbl_lbl = Label(self.metricsFrame, text="ES")
    # es_tbl_lbl.grid(row=0, column=2, padx=5, pady=5)

    # self.lv_ed_val = Label(self.metricsFrame, text="")
    # self.lv_ed_val.grid(row=1, column=1, padx=5, pady=5)

    # self.lvl_es_val = Label(self.metricsFrame, text="")
    # self.lvl_es_val.grid(row=1, column=2, padx=5, pady=5)

    # self.dice_ed = Label(self.metricsFrame, text="")
    # self.dice_ed.grid(row=2, column=1, padx=5, pady=5)
    # self.dice_es = Label(self.metricsFrame, text="")
    # self.dice_es.grid(row=2, column=2, padx=5, pady=5)

    # self.vol_ed = Label(self.metricsFrame, text="")
    # self.vol_ed.grid(row=3, column=1, padx=5, pady=5)
    # self.vol_es = Label(self.metricsFrame, text="")
    # self.vol_es.grid(row=3, column=2, padx=5, pady=5)

    # self.ef_lbl = Label(self.metricsFrame, text="")
    # self.ef_lbl.grid(row=4, column=1, padx=5, pady=5)

    # self.next_patient_btn = Button(self.metricsFrame, text=">", command=self.next_patient)
    # self.next_patient_btn.grid(row=6, column=1)

    # self.prev_patient_btn = Button(self.metricsFrame, text="<", command=self.prev_patient)
    # self.prev_patient_btn.grid(row=6, column=0)

  def open_path_dialog(self):
    print("open path")

  def next_frame(self):
    if (self.slice_index > 0):
      self.slice_index = self.slice_index - 1
      self.process()

  def prev_frame(self):
    if (self.slice_index < self.total_frames):
      self.slice_index = self.slice_index + 1
      self.process()

  def preprocess_image(self, img, fname = "temp.png"):
    resized = resize(img, (384, 384))
    imsave(fname, resized)
    return imread(fname)

  def predict(self):
    ## normalising prediction image
    norm_image = cv2.normalize(self.frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image  = np.expand_dims(norm_image, axis=-1)
    norm_image  = np.expand_dims(norm_image, axis=0)

    preds_train = self.model.predict(norm_image, verbose=1)
    self.prediction_raw = preds_train*255

    best_dice = self.getbest_dice(self.prediction_raw, self.mask)
    itemindex= best_dice[90:255].argmax() + 90

    preds_perfect = (self.prediction_raw > itemindex-1).astype(np.bool)
    preds_perfect = preds_perfect[...,3].squeeze()

    ## predicted mask from model
    self.prediction = preds_perfect
    return preds_perfect

  def process_plot(self, filename):
    plot_img = imread(filename)
    plot_img = resize(plot_img, (384, 384))
    rescaled_image = 255 * plot_img
    final_image = rescaled_image.astype(np.uint8)
    return final_image

  def roc(self):
    y_mask = self.mask_raw
    y_mask = tf.keras.utils.to_categorical(y_mask, num_classes=4, dtype='bool')
    y_covid = y_mask[...,self.axis].squeeze()

    y_predicted = self.prediction
    # we want to make them into vectors
    ground_truth_labels = y_covid.ravel()
    score_value= y_predicted.ravel()
    fpr, tpr, _ = roc_curve(ground_truth_labels, score_value)
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
    plt.savefig('roc.png')
    return 'roc.png'

  ## this function cycles pixel intensity to get the best dice coefficient
  def getbest_dice(self, preds_train_func,pred_mask):
    axis = 3
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

  def heatmap(self):
    # heat map to show prediction in a greater detail
    heatmask=preds_train255[...,axis].squeeze()
    return heatmask

  def init(self):
    tr_im = nib.load('./cases/coronacases_002.nii.gz')
    tr_masks = nib.load('./masks/coronacases_002.nii.gz')
    self.im_data = tr_im.get_fdata()
    self.mask_data = tr_masks.get_fdata()
    self.total_frames = self.im_data.shape[2]
    self.process()

  def process(self):
    slice_1 = self.im_data[:,:,self.slice_index]
    mask_1 = self.mask_data[:,:,self.slice_index]

    self.frame = self.preprocess_image(slice_1, 'frame.png')

    self.mask_raw = resize(mask_1, (384, 384))
    self.mask = (self.mask_raw == self.axis).astype(np.bool)

    pred = self.predict()
    roc_fname = self.roc()
    heatmap = self.heatmap()
    self.set_image(self.org_mask_cmp, self.create_image_component(self.mask))
    self.set_image(self.pred_mask_cmp, self.create_image_component(pred))
    roc_image = self.process_plot(roc_fname)
    self.set_image(self.analysis_cmp, self.create_image_component(roc_image))
    self.set_image(self.heatmask_cmp, self.create_image_component(heatmap))

  def place_ui_images(self):
    img_cmp = self.create_image_component(self.mask)
    self.set_image(self.org_mask_cmp, img_cmp)

  def dice_coef(self, y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def load_model(self):
    self.model = tf.keras.models.load_model("model", custom_objects={'dice_coef': self.dice_coef})

  def create_image_component(self, data):
    # load = Image.open(data)
    return ImageTk.PhotoImage(image=Image.fromarray(data))

  def set_image(self, dstFrame, image):
    dstFrame.configure(image=image)
    dstFrame.image = image

if __name__ == '__main__':
  # matplotlib.use("Agg")
  root = Tk()
  app = App(root)

  root.mainloop()