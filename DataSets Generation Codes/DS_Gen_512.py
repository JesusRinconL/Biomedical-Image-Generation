import numpy as np
import os
from PIL import ImageOps, Image
import random
import pickle
import re

def randomOrigins(imageHeight, imageWidth, sampleHeight, sampleWidth, minDistance, Nsamples):
    AUX = []
    while len(AUX) <Nsamples:
        #print(len(AUX))
        # Generar números aleatorios entre 0 y N
        N = random.randint(0, imageHeight - sampleHeight)
        M = random.randint(0, imageWidth - sampleWidth)
        # Crear el nuevo vector
        new_vector = np.asarray([N, M])

        # Comprobar la distancia con todos los vectores en AUX
        distancesOK = all(np.linalg.norm(new_vector - vector) > minDistance for vector in AUX)

        # Si la distancia es válida, agregar el nuevo vector a AUX
        if distancesOK:
            AUX.append(new_vector)

    return np.asarray(AUX)

if __name__ == '__main__':

    HEIGHT_image = 1536
    WIDTH_image = 2048

    HEIGHT = 512
    WIDTH = 512

    NUM_labels = 5
    LABELS = np.linspace(0, 255, NUM_labels, dtype=np.uint8)


    # 0 HEALING
    # 1 TISSUE_BACKGROUND
    # 2 CYTO
    # 3 BOUNDUARY CYTO
    # 4 NUCLEI

    main_dir = "C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO2\\GenerarDatasets"
    image_directory = "C:\\Users\\Jesus\\Documents\\TFG\\datasets_segmentado\\images"
    image_LABELdirectory = os.path.join(main_dir, "datasets_patrones\\DS1__4capas_dilatado")
    dataset_path = os.path.join(main_dir,"DataSet__4capas_512_dilatado")
    path_test = os.path.join(dataset_path,"test")
    path_train = os.path.join(dataset_path,"train")
    path_val = os.path.join(dataset_path,"val")


    used_images_file = os.path.join(os.path.dirname(main_dir), "used_images\\used_images5.plk")


    used_images = []
    #TODO: para guardar la lista
    #with open(used_images_file, "wb") as f:
    #    pickle.dump(used_images, f)

    if os.path.exists(used_images_file):
        # Cargar la lista desde el archivo
        with open(used_images_file, "rb") as f:
            used_images = pickle.load(f)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
        print(f"Directory '{dataset_path}' created.")
    else:
        print(f"Directory '{dataset_path}' already exists.")

    if not os.path.exists(path_test):
        os.makedirs(path_test)
        print(f"Directory '{path_test}' created.")
        contTEST = 1
    else:
        image_files = sorted(os.listdir(path_test))
        test_files = [file for file in image_files if file.lower().endswith(".bmp")]
        if len(test_files)>0:
            fileNumbers = [int(re.search(r'\d+', nombre).group()) for nombre in test_files]
            contTEST = np.max(np.asarray(fileNumbers)) + 1
        else:
            contTEST = 1
        print(f"Directory '{path_test}' already exists, and contains {contTEST-1} samples")

    if not os.path.exists(path_train):
        os.makedirs(path_train)
        print(f"Directory '{path_train}' created.")
        contTRAIN = 1
    else:
        image_files = sorted(os.listdir(path_train))
        test_files = [file for file in image_files if file.lower().endswith(".bmp")]
        if len(test_files) > 0:
            fileNumbers = [int(re.search(r'\d+', nombre).group()) for nombre in test_files]
            contTRAIN = np.max(np.asarray(fileNumbers)) + 1
        else:
            contTRAIN = 1
        print(f"Directory '{path_train}' already exists, and contains {contTRAIN-1} samples")

    if not os.path.exists(path_val):
        os.makedirs(path_val)
        print(f"Directory '{path_val}' created.")
        contVAL = 1
    else:
        image_files = sorted(os.listdir(path_val))
        test_files = [file for file in image_files if file.lower().endswith(".bmp")]
        if len(test_files) > 0:
            fileNumbers = [int(re.search(r'\d+', nombre).group()) for nombre in test_files]
            contVAL = np.max(np.asarray(fileNumbers)) + 1
        else:
            contVAL = 1
        print(f"Directory '{path_val}' already exists, and contains {contVAL-1} samples")


    # Obtener la lista de archivos en el directorio
    image_files = sorted(os.listdir(image_directory))
    jpg_files = [file for file in image_files if file.lower().endswith(".jpg")]

    for ii in range(len(jpg_files)):
        image_file = os.path.join(image_directory, jpg_files[ii])
        labelsMASKfile = os.path.join(image_LABELdirectory, jpg_files[ii].rsplit('.', 1)[0] + ".npy")

        if not (jpg_files[ii] in used_images):

            if os.path.exists(image_file) & os.path.exists(labelsMASKfile):

                gray_image = Image.open(image_file).convert("L")
                autocontrast_image = ImageOps.autocontrast(gray_image)
                autocontrastI = np.asarray(autocontrast_image)

                MaskLabels = np.load(labelsMASKfile)
                MaskImage = LABELS[MaskLabels]

                Origins = randomOrigins(HEIGHT_image, WIDTH_image, HEIGHT, WIDTH, np.linalg.norm([HEIGHT, WIDTH]) * 0.5, 6)


                # Train set
                for jj in range(4):
                    N, M = Origins[jj, :]
                    imageSAMPLE = autocontrastI[N:N + HEIGHT, M:M + WIDTH]
                    labelSAMPLE = MaskImage[N:N + HEIGHT, M:M + WIDTH]
                    sample = np.hstack(( imageSAMPLE, labelSAMPLE))
                    imagen_pillow = Image.fromarray(sample, mode='L')
                    imagen_pillow.save(os.path.join(path_train, f"{jj + contTRAIN}.bmp"))

                contTRAIN = contTRAIN + 4


                # Validation set
                for jj in range(3,4):
                    N, M = Origins[jj, :]
                    imageSAMPLE = autocontrastI[N:N + HEIGHT, M:M + WIDTH]
                    labelSAMPLE = MaskImage[N:N + HEIGHT, M:M + WIDTH]
                    sample = np.hstack((imageSAMPLE, labelSAMPLE))
                    imagen_pillow = Image.fromarray(sample, mode='L')
                    imagen_pillow.save(os.path.join(path_val, f"{jj - 3 + contVAL}.bmp"))

                contVAL = contVAL + 1

                #Test set
                for jj in range(4,5):
                    N, M = Origins[jj, :]
                    imageSAMPLE = autocontrastI[N:N + HEIGHT, M:M + WIDTH]
                    labelSAMPLE = MaskImage[N:N + HEIGHT, M:M + WIDTH]
                    sample = np.hstack((imageSAMPLE, labelSAMPLE))
                    imagen_pillow = Image.fromarray(sample, mode='L')
                    imagen_pillow.save(os.path.join(path_test, f"{jj - 4 + contTEST}.bmp"))

                contTEST = contTEST + 1

                used_images.append(jpg_files[ii])
                with open(used_images_file, "wb") as f:
                    pickle.dump(used_images, f)
                print("Samples extracted from: " + jpg_files[ii])