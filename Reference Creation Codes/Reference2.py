import matplotlib

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
from scipy.ndimage import binary_dilation, binary_erosion
from PIL import ImageOps, Image

def GenerateMaskImage_N(masks, Nlabels):
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink', 'teal',
              'lavender', 'maroon', 'olive', 'navy']


    colors = (colors * ((Nlabels + len(colors) - 1) // len(colors)))[:Nlabels]

    cmap = ListedColormap(colors)
    color_matrix = cmap(np.arange(cmap.N))

    masks2 = masks % len(colors)
    color_matrix = color_matrix[:, :-1]

    colored_image = color_matrix[masks2]

    ALPHA= np.ones_like(colored_image[:, :, 0]) * 0.25
    colored_masks = np.dstack((colored_image, ALPHA))

    return colored_masks


# 0 HEALING
# 1 TISSUE_BACKGROUND
# 2 CYTO_Layer1
# 3 CYTO_Layer2
# 4 CYTO_Layer3
# 5 BOUNDUARY CYTO
# 6 NUCLEI
# 7 BOUNDUARY NUCLEI

main_dir = "C:\\Users\\Jesus\\Documents\\TFG\\datasets_segmentado"

image_directory = os.path.join(main_dir, "images") 
image_CYTOdirectory = os.path.join(main_dir , "cytoMASKS")
image_NUCLEIdirectory =os.path.join(main_dir , "nucleiMASKS")
image_HEALINGdirectory = os.path.join(main_dir , "healing")
image_CYTO_Bond_directory = os.path.join(main_dir , "cytoMASKS_bond")
image_NUCLEI_Bond_directory = os.path.join(main_dir, "nucleiMASKS_bond")

# CARPETA DE SALIDA
image_LABELdirectory = "C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO2\\GenerarDatasets\\datasets_patrones\\DS1__6capas_Vir"

if not os.path.exists(image_LABELdirectory):
    os.makedirs(image_LABELdirectory)
    print(f"Directory '{image_LABELdirectory}' created.")
else:
    print(f"Directory '{image_LABELdirectory}' already exists.")

# Obtener la lista de archivos en el directorio
image_files = sorted(os.listdir(image_directory))
jpg_files = [file for file in image_files if file.lower().endswith(".jpg")]

ejecutado_capas = False
ejecutado_pintar = False
structure_element = np.ones((3, 3)) # Tamaño del kernel de la erosión y dilatación

for ii in range(len(jpg_files)):

    labelsMASKfile = os.path.join(image_LABELdirectory, jpg_files[ii].rsplit('.', 1)[0] + ".npy")
    healingMASKfile = os.path.join(image_HEALINGdirectory, jpg_files[ii].rsplit('.', 1)[0] + ".npy")
    cytoMASKfile =  os.path.join(image_CYTOdirectory, jpg_files[ii].rsplit('.', 1)[0] + ".npy")
    nucleiMASKfile = os.path.join(image_NUCLEIdirectory, jpg_files[ii].rsplit('.', 1)[0] + ".npy")

    if not os.path.exists(labelsMASKfile):
        if os.path.exists(healingMASKfile) & os.path.exists(nucleiMASKfile) & os.path.exists(cytoMASKfile):

            masksCYTO = np.load(cytoMASKfile)
            masksNUCLEI = np.load(nucleiMASKfile)
            maskHEALING = np.load(healingMASKfile)

            gray_image = Image.open(os.path.join(image_directory, jpg_files[ii])).convert("L")
            autocontrast_image = ImageOps.autocontrast(gray_image)
            autocontrastI = np.asarray(autocontrast_image)

            boundary_maskCYTO = np.zeros_like(maskHEALING)
            print("Calculando las fronteras de los citoplasmas")
            step_size = max(1, np.max(masksCYTO) // 100)
            for i in range(1, np.max(masksCYTO)):
                maskLABEL = np.zeros_like(maskHEALING)
                maskLABEL[masksCYTO == i] = True
                boundary_maskCYTO = boundary_maskCYTO | (maskLABEL & ~binary_erosion(maskLABEL))
                if i % step_size == 0:
                    print('.', end='', flush=True)
            print()
            boundary_maskNUCLEI = np.zeros_like(maskHEALING)
            print("Calculando las fronteras de los nucleos")
            step_size = max(1, np.max(masksNUCLEI) // 100)
            for i in range(1, np.max(masksNUCLEI)):
                maskLABEL = np.zeros_like(maskHEALING)
                maskLABEL[masksNUCLEI == i] = True
                boundary_maskNUCLEI = boundary_maskNUCLEI | (maskLABEL & ~binary_erosion(maskLABEL))
                if i % step_size == 0:
                    print('.', end='', flush=True)
            print()

            # PARA DILATAR LAS FRONTERAS DE LOS CITOPLASMAS
            # Comentar si se quiere patrón sin dilatar
            # boundary_maskCYTO = binary_dilation(boundary_maskCYTO, structure_element)
            
            cytoMASK2 = masksCYTO.copy()
            cytoMASK2[masksNUCLEI != 0] = 0

            labels = np.unique(cytoMASK2)
            labels = labels[labels != 0]

            # Inicializar un array para almacenar los valores medios por región
            label_mean = np.zeros(len(labels))

            # Calcular el valor medio para cada región
            for j, label in enumerate(labels):
                region = autocontrastI[cytoMASK2 == label]
                label_mean[j] = np.mean(region)
            
            if not ejecutado_capas:
                # Ordena las medias de menor a mayor
                sorted_means = np.sort(label_mean)
                # Divide el rango de medias en tres partes iguales
                lower_bound = sorted_means[int(len(sorted_means) / 3)]
                upper_bound = sorted_means[int(2 * len(sorted_means) / 3)]
                ejecutado_capas = True
                print("EL valor del límite inferior es", round(lower_bound, 2), "y del límite superior", round(upper_bound, 2))
                
            # Utiliza los límites para clasificar las etiquetas en tres capas
            labels1stLayer = labels[label_mean < lower_bound]
            labels2ndLayer = labels[(label_mean >= lower_bound) & (label_mean < upper_bound)]
            labels3rdLayer = labels[label_mean >= upper_bound]
            
            cytoMASK1stLayer = cytoMASK2.copy()
            cytoMASK1stLayer[np.isin(cytoMASK1stLayer, np.hstack((labels2ndLayer,labels3rdLayer)))] = 0

            cytoMASK2ndLayer = cytoMASK2.copy()
            cytoMASK2ndLayer[np.isin(cytoMASK2ndLayer, np.hstack((labels1stLayer,labels3rdLayer)))] = 0

            cytoMASK3rdLayer = cytoMASK2.copy()
            cytoMASK3rdLayer[np.isin(cytoMASK3rdLayer, np.hstack((labels1stLayer, labels2ndLayer)))] = 0


            AUX = np.zeros_like(cytoMASK2)
            AUX[cytoMASK1stLayer != 0] = 1
            AUX[cytoMASK2ndLayer != 0] = 2

            imageLABELS = np.zeros_like(masksCYTO)
            imageLABELS[~maskHEALING] = 1
            imageLABELS[(cytoMASK1stLayer != 0) & (~boundary_maskCYTO)] = 2
            imageLABELS[(cytoMASK2ndLayer != 0) & (~boundary_maskCYTO)] = 3
            imageLABELS[(cytoMASK3rdLayer != 0) & (~boundary_maskCYTO)] = 4
            imageLABELS[(boundary_maskCYTO != 0) & (~maskHEALING)] = 5
            imageLABELS[(masksNUCLEI != 0) & (~boundary_maskNUCLEI)] = 6
            imageLABELS[boundary_maskNUCLEI] = 7
            np.save(labelsMASKfile, imageLABELS)

            if not ejecutado_pintar:
                NUM_labels = 5
                MasksColors = GenerateMaskImage_N(imageLABELS, NUM_labels)
                fig = plt.figure()
                fig.clear()  # Limpia la figura antes de volver a dibujar
    
                ax = fig.add_subplot(121)
                ax.clear()
                ax.imshow(imageLABELS, cmap='tab10')
                ax.set_title(f"{ii} para el fichero {jpg_files[ii].rsplit('.', 1)[0]}")
                # ax.set_axis_off()
    
                ax1 = fig.add_subplot(122)
                ax1.clear()
                ax1.imshow(autocontrastI, cmap='gray')
                ax1.set_title(f"{ii} para el fichero {jpg_files[ii].rsplit('.', 1)[0]}")
                ax1.imshow(MasksColors)
                # ax1.set_axis_off()
    
                fig.tight_layout()  # Llama a tight_layout después de haber añadido los subgráficos
                plt.show()
                # fig.canvas.draw()  # Esto fuerza a la actualización del lienzo
                # figureD.show()
                # plt.pause(0.1)
                ejecutado_pintar = True

        else:
            print("Faltan ficheros previos para procesar: " + jpg_files[ii].rsplit('.', 1)[0])
    else:
        print("El fichero de labels para " + jpg_files[ii].rsplit('.', 1)[0] + " ya existe")
