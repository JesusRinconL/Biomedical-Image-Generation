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

    ALPHA = np.ones_like(colored_image[:, :, 0]) * 0.25
    colored_masks = np.dstack((colored_image, ALPHA))

    return colored_masks

NUM_labels = 5
# 0 HEALING
# 1 TISSUE_BACKGROUND
# 2 CYTO
# 3 BONDUARY CYTO
# 4 NUCLEI

t = TicToc() #create instance of class
t.tic() #Start timer

main_dir = "C:\\Users\\Jesus\\Documents\\TFG\\datasets_segmentado"

image_directory = os.path.join(main_dir, "images") 
image_CYTOdirectory = os.path.join(main_dir , "cytoMASKS")
image_NUCLEIdirectory =os.path.join(main_dir , "nucleiMASKS")
image_HEALINGdirectory = os.path.join(main_dir , "healing")
image_CYTO_Bond_directory = os.path.join(main_dir , "cytoMASKS_bond")
image_NUCLEI_Bond_directory = os.path.join(main_dir, "nucleiMASKS_bond")

# CARPETA DE SALIDA
image_LABELdirectory = "C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO2\\GenerarDatasets\\datasets_patrones\\DS1__4capas_dilatado"


if not os.path.exists(image_LABELdirectory):
    os.makedirs(image_LABELdirectory)
    print(f"Directory '{image_LABELdirectory}' created.")
else:
    print(f"Directory '{image_LABELdirectory}' already exists.")
    
image_files = sorted(os.listdir(image_directory))
jpg_files = [file for file in image_files if file.lower().endswith(".jpg")]
ejecutado = False
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
            boundary_maskCYTO = binary_dilation(boundary_maskCYTO, structure_element)
            
            imageLABELS = np.zeros_like(masksCYTO)
            imageLABELS[~maskHEALING] = 1
            imageLABELS[(masksCYTO != 0) & (~boundary_maskCYTO)] = 2
            imageLABELS[(boundary_maskCYTO != 0) & (~maskHEALING)] = 3
            imageLABELS[(masksNUCLEI != 0) & (~boundary_maskNUCLEI)] = 4
            np.save(labelsMASKfile, imageLABELS)

            # PARA EJECUTAR LA REPRESENTACIÓN SOLO UNA VEZ
            if not ejecutado:
                gray_image = Image.open(os.path.join(image_directory, jpg_files[ii])).convert("L")
                autocontrast_image = ImageOps.autocontrast(gray_image)
                autocontrastI = np.asarray(autocontrast_image)
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

                # fig.tight_layout()  # Llama a tight_layout después de haber añadido los subgráficos
                # fig.canvas.draw()  # Esto fuerza a la actualización del lienzo
                # plt.pause(0.1)
                plt.show()
                ejecutado = True

        else:
            print("Faltan ficheros previos para procesar: " + jpg_files[ii].rsplit('.', 1)[0])
    else:
        print("El fichero de labels para " + jpg_files[ii].rsplit('.', 1)[0] + " ya existe")
