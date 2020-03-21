# Desafio GigaCandanga

Esse código foi feito usando Python3.6

## Dependencias

* OpenCV

    ```
    python3 -m pip install opencv-contrib-python==3.4.2.16
    ```

* Numpy

    ```
    python3 -m pip install numpy
    ```

* Scipy 

    ```
    python3 -m pip install scipy
    ```

* Scikit Learn

    ```
    python3 -m pip install sklearn
    ```

* Matplotlib

    ```
    python3 -m pip install matplotlib
    ```

* Seaborn

    ```
    python3 -m pip install seaborn
    ```

* Pickle

    ```
    python3 -m pip install pickle
    ```
* Pandas

    ```
    python3 -m pip install pandas
    ```

## Como executar

O código de reconhecimento de objetos recebe como argumento o nome da imagem de teste. Essa imagem deve estar presente em /dataset/GP. Foram disponibilizadas 4 imagens de teste (GP.png, GP2.png, GP3.png, GP4.png)

```
python3 object_recognition.py GP.png
```

O código de classificação de objetos recebe como argumento os parâmetros:

1. --train ou --test
2. --orb ou --sift
3. Nome do arquivo para salvar o classificador

Lembrando que ao fazer o teste de um classificador o arquivo passado como parâmetro deve ter sido treinado com o mesmo extrator de features passado como parâmetro

Para treinar:

```
python3 object_classifier.py --train --orb clf.sav
```

Para testar:

```
python3 object_classifier.py --test --orb clf.sav
```

