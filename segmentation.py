import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
from cellpose import models
from cellpose.io import imread
import numpy as np
import os
from scipy.ndimage import binary_erosion
from PIL import ImageOps, Image
import cv2

def circular_kernel(radius):
    """Create a circular kernel with the given radius."""
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    return kernel

def circular_erode(image, radius):
    """Erode an image using a circular kernel."""
    kernel = circular_kernel(radius)
    result = cv2.erode(image, kernel)
    return result

def circular_dilate(image, radius):
    """Dilate an image using a circular kernel."""
    kernel = circular_kernel(radius)
    result = cv2.dilate(image, kernel)
    return result

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modelCYTO = models.Cellpose(gpu=False, model_type='cyto')
    modelNUCLEI = models.Cellpose(gpu=False, model_type='nuclei')

    # Path to the image directory
    image_directory = "/path/to/image/directory"
    # image_CYTOdirectory =  "/path/to/image/Cytoplasm/directory"
    # image_NUCLEIdirectory = "/path/to/image/Nuclei/directory"
    # imagesSET_256 = "/path/to/image/healing256x256/directory"
    # imagesSET_1024_2048 = "/path/to/image/healing1024x2048/directory"

    # Get the list of files in the directory
    image_files = sorted(os.listdir(image_directory))
    jpg_files = [file for file in image_files if file.lower().endswith(".jpg")]

    NUM_labels = 5
    LABELS = np.linspace(0, 255, NUM_labels, dtype=np.uint8)

    fig = plt.figure()

    for ii in range(len(jpg_files)):
        gray_image = Image.open(os.path.join(image_directory, jpg_files[ii])).convert("L")
        autocontrast_image = ImageOps.autocontrast(gray_image)
        autocontrastI = np.asarray(autocontrast_image)

        N, M = autocontrastI.shape

        masksCYTO, flowsCYTO, stylesCYTO, diamsCYTO = modelCYTO.eval(np.array(autocontrast_image), flow_threshold=0.9, diameter=31.1, channels=[0, 0])
        print("CYTO masks calculated")
        masksNUCLEI, flowsNUCLEI, stylesNUCLEI, diamsNUCLEI = modelNUCLEI.eval(np.array(autocontrast_image), diameter=15.2, channels=[0, 0])
        print("NUCLEI masks calculated")

        # Now we force each CYTO to be surrounded by 0
        boundary_mask = np.zeros((N, M), dtype=np.bool_)

        for i in range(1, np.max(masksCYTO)):
            maskLABEL = np.zeros((N, M), dtype=np.bool_)
            maskLABEL[masksCYTO == i] = True
            boundary_mask = boundary_mask | (maskLABEL & ~binary_erosion(maskLABEL))
        
        print("CYTO boundaries calculated")
        # We will divide the scar and tissue based on the CYTO detection.
        healingZone_mask = np.zeros((N, M), dtype=np.bool_)
        healingZone_mask[masksCYTO == 0] = True

        frame_width = 60
        new_h, new_w = N + 2 * frame_width, M + 2 * frame_width
        matrix_with_frame = np.zeros((new_h, new_w), dtype=bool)
        matrix_with_frame[frame_width:new_h - frame_width, frame_width:new_w - frame_width] = healingZone_mask

        matrix_with_frame = matrix_with_frame.astype(np.uint8) * 255

        eroded_matrix = circular_erode(matrix_with_frame, 60)
        dilated_matrix = circular_dilate(eroded_matrix, 140)
        dilated_matrix2 = circular_erode(dilated_matrix, 50)

        healingZone_mask2 = dilated_matrix[frame_width:new_h - frame_width, frame_width:new_w - frame_width] > 1
        healingZone_mask3 = dilated_matrix2[frame_width:new_h - frame_width, frame_width:new_w - frame_width] > 1
        healingZone_mask = healingZone_mask2 & (masksCYTO == 0) & (masksNUCLEI == 0)

        boundary_mask2 = (boundary_mask & ~healingZone_mask3)

        cyto_mask = masksCYTO != 0
        cyto_mask = cyto_mask & ~boundary_mask2
        nuclei_mask = masksNUCLEI != 0
        tissue_mask = ~healingZone_mask & ~nuclei_mask & ~cyto_mask

        label_mask = np.zeros((N, M), dtype=np.uint8)
        label_mask[~healingZone_mask] = LABELS[1]
        label_mask[cyto_mask] = LABELS[2]
        label_mask[nuclei_mask] = LABELS[3]
        label_mask[tissue_mask] = LABELS[4]

        # fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # ax.imshow(label_mask,  cmap='viridis')

        image256_img = autocontrastI[512+125:512+125+256, 700:700+256]
        image256_label = label_mask[512+125:512+125+256, 700:700+256]
        image256 = np.hstack((image256_img, image256_label))
        imagen_pillow = Image.fromarray(image256, mode='L')
        imagen_pillow.save(os.path.join(imagesSET_256, f"{ii + 1}.jpg"))

        image1024_img = autocontrastI[0:1024, 0:2048]
        image1024_label = label_mask[0:1024, 0:2048]
        image1024 = np.hstack((image1024_img, image1024_label))
        imagen_pillow = Image.fromarray(image1024, mode='L')
        imagen_pillow.save(os.path.join(imagesSET_1024_2048, f"{ii + 1}.jpg"))
        fig.clear()  # Clear the figure before redrawing

        ax = fig.add_subplot(111)
        ax.imshow(image1024, cmap='gray')
        ax.set_title(f"{ii}")
        fig.tight_layout()  # Call tight_layout after adding subplots
        fig.canvas.draw()  # Force canvas update
        plt.pause(0.01)
        print("Moving to the next image")

    plt.show()
