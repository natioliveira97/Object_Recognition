# -*- coding: utf-8 -*-

import sys
import os
import numpy as np
from scipy import stats
import pandas as pd
import time

import cv2

import pickle

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import seaborn as sns


folder_path = "../dataset/"
classes = ["Bateria", "Cooler", "HD_SSD", "Pente_RAM", "Placa_de_Video", "Processador"]

# Carrega imagens de uma pasta
def load_images_from_folder(folder):

    images = []

    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    
    images = np.array(images)

    # Retorna array de imagens
    return images

# Seleciona os dados de treinamento e teste aleatoriamente
def data_selection(folder_path, classes, train_proportion): 

    train_images = [] 
    train_labels = [] 
    test_images = []
    test_labels = []

    for i in range(len(classes)):
        # Carrega imagens da pasta
        images = load_images_from_folder((folder_path+classes[i]))

        random_index = np.random.choice(len(images), size=int(len(images)*train_proportion), replace=False)
        
        train_images = train_images + images[random_index].tolist()
        train_labels = train_labels + [classes[i]]*int(len(images)*train_proportion)

        test_images = test_images + np.delete(images,random_index).tolist()
        test_labels = test_labels + [classes[i]]*int(len(images)*(1-train_proportion))

    return train_images, train_labels, test_images, test_labels

# Extrai features de imagem
def extract_features(image, feature_method):

    # Escolhe o extrator de features
    fm_dict = {'--orb': cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=500),
               '--sift': cv2.xfeatures2d.SIFT_create(),
               '--surf': cv2.xfeatures2d.SURF_create()}

    # Transforma imagem em grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    if feature_method in fm_dict:
        f_method = fm_dict[feature_method]
    else:
        print("Only suports orb, sift and surf descriptors")
        exit()

    # Detecta keypoints e descritores
    kp = f_method.detect(gray, None)
    kp, des = f_method.compute(gray, kp) 

    # Transforma descritores em array
    des = np.array(des)

    return des

# Treinamento do método de aprendizado
def train_data(train_images, train_label, filename, feature_method):
    print("Extraindo features das imagens ...")

    des_list = []
    label_list = []

    # Para cada imagem, retira os descritores da imagem e coloca na lista de descritores
    for j in range(len(train_images)):
        des = extract_features(train_images[j], feature_method)
        if(des.shape!=()):
            des_list = des_list + des.tolist()
            label_list = label_list + [train_label[j]]*len(des)
    
    print("Inicio do treinamento ...")
    t0 = time.time()
    clf = OneVsOneClassifier(svm.SVC(class_weight = 'balanced', gamma='scale'), n_jobs = -1)
    clf.fit(des_list, label_list)
    t1 = time.time()
    print("O treinamento demorou", t1-t0, " segundos")
    pickle.dump(clf, open((filename), 'wb'))

# Plota os gráficos de matriz de confusão
def plot_confusion_matrix(y_true, y_pred, classes, title):
    acc = accuracy_score(y_true, y_pred)
    title = title + u" (Acuracia: " + str("{:10.4f}".format(acc)) + ")"

    cm = confusion_matrix(y_true, y_pred, classes)
    cm = normalize(cm, axis=1, norm='l1')
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.ylabel(u'True label')
    plt.xlabel(u'Predicted label')
    plt.show()

# Teste do método de aprendizado
def test_data(test_images, test_labels, filename, feature_method):
    des_list = []
    pred_list = []

    # Carrega o classificador
    clf = pickle.load(open((filename), 'rb'))

    # Para cada imagem, retira os descritores orb da imagem e coloca na lista de descritores
    for j in range(len(test_images)):
        des = extract_features(test_images[j], feature_method)
        if(des.shape!=()):
            prediction = clf.predict(des)
            pred_list.append(stats.mode(prediction).mode[0])

    plot_confusion_matrix(test_labels, pred_list, classes, u"Matriz de Confusao")

def main():

    if(len(sys.argv)  < 4):
        print("Argumentos de entrada:")
        print("1. --train --test")
        print("2. --sift --orb")
        print("3. Nome do classificador")    
        exit()
    if sys.argv[1] == "--train":
        train_images, train_labels, test_images, test_labels = data_selection(folder_path, classes, 0.7)
        train_data(train_images, train_labels, sys.argv[3], sys.argv[2])

    if sys.argv[1] == "--test":
        train_images, train_labels, test_images, test_labels = data_selection(folder_path, classes, 0.7)
        test_data(test_images, test_labels, sys.argv[3], sys.argv[2])

main()






