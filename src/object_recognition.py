# -*- coding: utf-8 -*-

import sys
import os

import numpy as np
import cv2


class FindObjects:

    def __init__(self, classes, feature_method):
        self.feature_method = feature_method
        self.classes = classes
        self.kp = []
        self.des = []
        self.label = []
        self.images = []
        self.draw = {k: None for k in classes}
        self.must_have = set(classes)
    

    # Loads all images from folder
    def load_images_from_folder(self, folder_path):

        images = []
        # Load images in python list
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            if img is not None:
                images.append(img)

        return images

    def extract_features(self, image):
        # Feature method dictionary
        fm_dict = {'orb': cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE, nfeatures=500),
                'sift': cv2.xfeatures2d.SIFT_create(),
                'surf': cv2.xfeatures2d.SURF_create()}

        # Transform imagem in grayscale
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        if self.feature_method in fm_dict:
            f_method = fm_dict[self.feature_method]
        else:
            print("Only suports orb, sift and surf descriptors")
            exit()

        # Detect keypoins and the keypoints descriptors
        kp = f_method.detect(gray, None)
        kp, des = f_method.compute(gray, kp) 

        return kp, des

    def colect_data(self, folder_path):
        for i in range(len(self.classes)):
            images = self.load_images_from_folder(folder_path + self.classes[i])
            self.images = self.images + images

            for j in range(len(images)):
                kp, des = self.extract_features(images[j])
                self.kp.append(kp)
                self.des.append(des)
                self.label.append(self.classes[i])

        # Transform python list in np.array
        self.images = np.array(self.images)
         
    def find_objects(self, image_name):
        objects_found = []

        # Carrega imagem do Gigaponto
        gp_image = cv2.imread(image_name)
        if gp_image is None:
            print("Não foi possível abrir a imagem")
            exit()

        # Encontra os kepoints e descritores da imagem do Gigaponto
        gp_kp, gp_des = self.extract_features(gp_image)

        # Inicializa um BFMatcher, responsável por encontrar os pontos em comum entre as imagens
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)

        result_image = gp_image

        # Procura os objetos na imagem do Gigaponto
        for i in range(len(self.images)):

            # Encontra pontos em comum entre as imagens
            match = bf.knnMatch(self.des[i],gp_des, k=2)

            # Seleciona bons pontos em comum
            good_match = []
            for m,n in match:
                if m.distance < 0.75*n.distance:
                    good_match.append(m)

            if len(good_match) > 30:
                # Vetor de pontos na imagem do objeto
                src_pts = np.float32([self.kp[i][m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                # Vetor de pontos na imagem do Gigaponto que combinam com o objeto
                dst_pts = np.float32([gp_kp[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
                # Encontra a matriz de transformação entre as imagens
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
                # Dimensão do objeto
                h,w = self.images[i].shape[:2]
                # Bounding box do objeto
                box = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                # Aplica a matrix de homografia na bounding box do objeto
                dst = cv2.perspectiveTransform(box,M)
                objects_found.append((dst, self.label[i], len(good_match)))

        return objects_found

    def show_result(self, image_name, objects_found):

        # Carrega imagem do Gigaponto
        gp_image = cv2.imread(image_name)
        if gp_image is None:
            print("Não foi possível abrir a imagem")
            exit()

        # Examina se todos os objetos foram encontrados
        must_have = set(self.classes)
        found = set()
        for i in objects_found:
            if i[1] in found:
                if(self.draw[i[1]][2] < i[2]):
                    self.draw[i[1]] = i       
            else:
                found.add(i[1])
                self.draw[i[1]] = i

        if self.must_have == found:
            print("Esse Gigaponto possui todos os elementos.")
        
        else:
            print("Esse Gigaponto não possui os elementos: ", end = '')
            print(list(self.must_have-found))

        # Desenha box em volta do objeto encontrado
        for i in found:
            color = np.int64(np.random.randint(0,255, size=(3, )))
            color = (int(color[0]), int(color[1]), int(color[2]))
            result_image = cv2.polylines(gp_image, [np.int32(self.draw[i][0])], True, color, 5, cv2.LINE_AA)
            cv2.putText(gp_image, self.draw[i][1], tuple(self.draw[i][0][0][0]) , cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Mostra imagem
        cv2.namedWindow("Objetos Encontrados", cv2.WINDOW_NORMAL)  
        cv2.imshow("Objetos Encontrados", gp_image)
        cv2.waitKey()


def main():

    if(len(sys.argv)  < 2):
        print("Argumento de entrada:")
        print("1. Nome da imagem de teste (A imagem deve estar em ../dataset/GP/")   
        exit()

    folder_path = "../dataset/"
    classes = ["Bateria", "Cooler", "HD_SSD", "Pente_RAM", "Placa_de_Video", "Processador"]
    
    fo = FindObjects(classes, "sift")
    fo.colect_data(folder_path)
    objects_found = fo.find_objects(folder_path + "GP/" + sys.argv[1])
    fo.show_result(folder_path + "GP/" + sys.argv[1],objects_found)

main()






