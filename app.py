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
from skimage.io import imread, imshow
from skimage.transform import resize
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
    # self.predprob = 0.5
    # self.loaded_images = { "ES": None, "ED": None }
    # # Patient images in nibabel format full path
    # self.patient_images = {}
    # # Patient info in a dictionary
    # self.image_order = 0
    # GUI related
    self.master = master
    self.create_ui()
    self.create_sidebar()
    self.load_model()

  def create_ui(self):
    self.master.geometry("1020x570")
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

    # srcLbl = Label(self.imageFrame, text="ED Slice")
    # srcLbl.grid(row=0, column=0)

    # self.edimg = Label(self.imageFrame)
    # self.edimg.grid(row=1, column=0, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ED Mask")
    # srcLbl.grid(row=0, column=1)

    # self.edmask = Label(self.imageFrame)
    # self.edmask.grid(row=1, column=1, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ES Slice")
    # srcLbl.grid(row=2, column=0)

    # self.esimg = Label(self.imageFrame)
    # self.esimg.grid(row=3, column=0, padx=5, pady=5)

    # srcLbl = Label(self.imageFrame, text="ES Mask")
    # srcLbl.grid(row=2, column=1)

    # self.esmask = Label(self.imageFrame)
    # self.esmask.grid(row=3, column=1, padx=5, pady=5)

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

    # b = Button(self.imageControlsFrame, text=">", command=self.next_src_image)
    # b.grid(row=1, column=1, sticky=E)
    # b = Button(self.imageControlsFrame, text="<", command=self.prev_src_image)
    # b.grid(row=1, column=0, sticky=W)

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

  def dice_coef(self, y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

  def load_model(self):
    self.model = tf.keras.models.load_model("model", custom_objects={'dice_coef': self.dice_coef})  ## loading model

if __name__ == '__main__':
  # matplotlib.use("Agg")
  root = Tk()
  app = App(root)

  root.mainloop()