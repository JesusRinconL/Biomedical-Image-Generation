# This code is provided with .py but it is recommended to use .ipynb and select the code you need
import cv2
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnrimport torchvision.transforms as transforms
from scipy.ndimage import convolve
import lpips
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# Cargar imágenes
original_image = Image.open('C:\\path\\to\\image_real.png').convert('L')
generated_image = Image.open('C:\\path\\to\\image_real.png').convert('L')

# Pasar al mismo tamaño
original_image = original_image.resize((256, 256))
generated_image = generated_image.resize((256, 256))

# Convertir imágenes PIL a matrices NumPy
original_array = np.array(original_image)
generated_array = np.array(generated_image)

# Run only once
# Inicializar el modelo LPIPS
execute = False
if not execute:
  loss_fn = lpips.LPIPS(net='vgg')
  execute = True

def calculate_metrics(original_image, generated_image):
    ## Calcular PSNR
    psnr_value = psnr(original_array, generated_array)
    ## Calcular SSIM
    ssim_value = ssim(original_array, generated_array, win_size=7)
    ## Calcular MSE
    mse = mean_squared_error(original_array.flatten(), generated_array.flatten())
    ## Calcular MAE
    mae = mean_absolute_error(original_array.flatten(), generated_array.flatten())
    
    ## Calcular LPIPS
    ## CÓDIGO NECESARIO PARA CALCULAR LPIPS ##
    # Transformaciones necesarias para las imágenes
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]) # Modificar en función tal tamaño empleado
    # Aplicar transformaciones a las imágenes
    original_img = transform(original_image)
    generated_img = transform(generated_image)
    # Preparar las imágenes para la entrada en la métrica LPIPS
    original_img = original_img.unsqueeze(0)
    generated_img = generated_img.unsqueeze(0)
    
    # Calcular el LPIPS entre las imágenes
    lpips_value = loss_fn(original_img, generated_img).item()

    ## Calcular correlación de Pearson
    pearson_corr = np.corrcoef(original_array.flatten(), generated_array.flatten())[0, 1]

    ## Calcular entropia cruzada
    # Calcular histogramas normalizados de las imágenes
    hist_original, _ = np.histogram(original_array.flatten(), bins=256, range=[0, 256], density=True)
    hist_generated, _ = np.histogram(generated_array.flatten(), bins=256, range=[0, 256], density=True)
    epsilon = 1e-10
    hist_org = np.maximum(hist_original, epsilon)
    hist_gen = np.maximum(hist_generated, epsilon)
    cross_entropy = -np.sum(hist_org * np.log(hist_gen))
    
    ## Calcular la diferencia de histograma
    # Calcular histogramas
    hist_original_diff, _ = np.histogram(original_array.flatten(), bins=256, range=[0,256])
    hist_generated_diff, _ = np.histogram(generated_array.flatten(), bins=256, range=[0,256])
    # Normalizar histogramas
    hist_original_norm = hist_original_diff / np.sum(hist_original_diff)
    hist_generated_norm = hist_generated_diff / np.sum(hist_generated_diff)
    # Calcular distancia de Bhattacharyya
    hist_diff = np.sum(np.sqrt(hist_original_norm * hist_generated_norm))

    ## Calcular los errores de gradiente
    # Definir los kernels de Sobel para las derivadas X e Y
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    # Aplicar convolución para obtener gradientes
    grad_x_original = convolve(original_image, sobel_x)
    grad_y_original = convolve(original_image, sobel_y)
    grad_x_generated = convolve(generated_image, sobel_x)
    grad_y_generated = convolve(generated_image, sobel_y)
    # Calcular diferencia de gradientes
    grad_diff = np.mean(np.abs(grad_x_original - grad_x_generated) + np.abs(grad_y_original - grad_y_generated))

    
    return psnr_value, mse, mae, ssim_value, lpips_value, pearson_corr, cross_entropy, hist_diff, grad_diff

# Calcular métricas
psnr_value, mse, mae, ssim_value, lpips, pearson_corr, cross_entropy, hist_diff, grad_diff = calculate_metrics(original_image, generated_image)

# Imprimir resultados
print("PSNR:", round(psnr_value, 2))
print("SSIM:", round(ssim_value, 2))
print("MSE:", round(mse, 2))
print("MAE:", round(mae, 2))
print("LPIPS:", round(lpips, 2))
print("Correlacion de Pearson:", round(pearson_corr, 2))
print("Entropía cruzada:", round(cross_entropy, 2))
print("Diferencia de Histograma:", round(hist_diff, 2))
print("Errores de gradiente:", round(grad_diff, 2))

### This code is to save information of different experiments
pruebas = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10", "P11", "P12"]
medias_pruebas = []
CV_pruebas = []
# To calculate metrics for a complete dataset:


psnr_total = 0
mse_total = 0
mae_total = 0
ssim_total = 0
lpips_total = 0
pearson_corr_total = 0
cross_entropy_total = 0
hist_diff_total = 0
grad_diff_total = 0
contador = 0

psnr_values = []
mse_values = []
mae_values = []
ssim_values = []
lpips_values = []
pearson_corr_values = []
cross_entropy_values = []
hist_diff_values = []
grad_diff_values = []

for i in tqdm(range(1, 431)):
    # Comprobar que existe la imagen
    if not os.path.isfile(f"C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO5\\imagenes_sinteticas\\img_sint_6c_512_25%\\PANC-1_{i}.jpg"):
        continue
        
    # Cargar imágenes
    original_image = Image.open(f"C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO5\\imagenes_originales\\imagenes_autocontrastadas_PANC\PANC-1_{i}.jpg").convert('L')
    generated_image = Image.open(f'C:\\Users\\Jesus\\Documents\\TFG\\PROYECTO5\\imagenes_sinteticas\\img_sint_6c_512_25%\\PANC-1_{i}.jpg').convert('L')
    
    # Convertir imágenes PIL a matrices NumPy
    original_array = np.array(original_image)
    generated_array = np.array(generated_image)

    # Ejecutar la función
    psnr_value, mse, mae, ssim_value, lpips, pearson_corr, cross_entropy, hist_diff, grad_diff = calculate_metrics(original_image, generated_image)

    # Añade a la lista los valores de cada imagen
    psnr_values.append(psnr_value)
    mse_values.append(mse)
    mae_values.append(mae)
    ssim_values.append(ssim_value)
    lpips_values.append(lpips)
    pearson_corr_values.append(pearson_corr)
    cross_entropy_values.append(cross_entropy)
    hist_diff_values.append(hist_diff)
    grad_diff_values.append(grad_diff)

    # Contador de cuantas imágenes se han calculado
    contador += 1
    
    # Experiments to analize outliers
    '''
    if mse == 112.35043334960938:
        # Crear una figura con dos subgráficos
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Mostrar la primera imagen en el primer subgráfico
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Imagen original {i}')
        axes[0].axis('off')  # Ocultar los ejes
        
        # Mostrar la segunda imagen en el segundo subgráfico
        axes[1].imshow(generated_image, cmap='gray')
        axes[1].set_title(f'Imagen sintética {i}')
        axes[1].axis('off')  # Ocultar los ejes
        
        # Mostrar la figura con las dos imágenes
        plt.show()
     '''
    if pearson_corr == 0.7257716446314266:
        # Crear una figura con dos subgráficos
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Mostrar la primera imagen en el primer subgráfico
        axes[0].imshow(original_image, cmap='gray')
        axes[0].set_title(f'Imagen original {i}')
        axes[0].axis('off')  # Ocultar los ejes
        
        # Mostrar la segunda imagen en el segundo subgráfico
        axes[1].imshow(generated_image, cmap='gray')
        axes[1].set_title(f'Imagen sintética {i}')
        axes[1].axis('off')  # Ocultar los ejes
        
        # Mostrar la figura con las dos imágenes
        plt.show()


# Calcular las medias
psnr_media = sum(psnr_values) / contador
mse_media = sum(mse_values) / contador
mae_media = sum(mae_values) / contador
ssim_media = sum(ssim_values) / contador
lpips_media = sum(lpips_values) / contador
pearson_corr_media = sum(pearson_corr_values) / contador
cross_entropy_media = sum(cross_entropy_values) / contador
hist_diff_media = sum(hist_diff_values) / contador
grad_diff_media = sum(grad_diff_values) / contador

# Calcular la desviación típica
psnr_std = np.std(psnr_values)
mse_std = np.std(mse_values)
mae_std = np.std(mae_values)
ssim_std = np.std(ssim_values)
lpips_std = np.std(lpips_values)
pearson_corr_std = np.std(pearson_corr_values)
cross_entropy_std = np.std(cross_entropy_values)
hist_diff_std = np.std(hist_diff_values)
grad_diff_std = np.std(grad_diff_values)

# Métricas y valores
metricas = ['PSNR', 'MSE', 'MAE', 'SSIM', 'LPIPS', 'Correlacion de Pearson', 'Entropia cruzada', 'Diferencia de histograma', 'Errores de gradiente']
valores = [psnr_values, mse_values, mae_values, ssim_values, lpips_values, 
           pearson_corr_values, cross_entropy_values, hist_diff_values, grad_diff_values]

# Medias y desviaciones estándar
medias = [psnr_media, mse_media, mae_media, ssim_media, lpips_media, pearson_corr_media, cross_entropy_media, hist_diff_media, grad_diff_media]
desviaciones = [psnr_std, mse_std, mae_std, ssim_std, lpips_std, pearson_corr_std, cross_entropy_std, hist_diff_std, grad_diff_std]

# Calcula el coeficiente de variación de cada métrica
coeficientes_variacion = []
for i, j in zip(medias, desviaciones):
    cv = (j / i) * 100
    coeficientes_variacion.append(cv)

# Añadir las medias y CVs de la prueba actual a las listas de pruebas
# Comentar si ya se ha realizado la prueba o no se desee guardar
# medias_pruebas.append(medias)
# CV_pruebas.append(coeficientes_variacion)

# Añadir a las lista que incluye todos los valores para su posterior representación
valores.append(coeficientes_variacion)

# Imprimir resultados
print("Las métricas obtenidas de ", contador, "imágenes son:")
for i, j in zip(medias, metricas):
    i = round(i, 3)
    print(f'La media de {j} es: {i}')
    
print('-'*60)
for i, j in zip(desviaciones, metricas):
    i = round(i, 3)
    print(f'La desviación típica de {j} es: {i}')

print('-'*60)
for i, j in zip(coeficientes_variacion, metricas):
    i = round(i, 3)
    print(f'El coeficiente de variación de {j} es: {i}')

## Study of outliers:
maximo = max(cross_entropy_values)
indice = cross_entropy_values.index(maximo)
print(f"El índice de {maximo} es {indice}.")
minimo = min(pearson_corr_values)
indice = pearson_corr_values.index(minimo)
print(f"El índice de {minimo} es {indice}.")

# Data Representation
# Métricas y datos para valores altos
metricas_valores_altos = ['PSNR', 'MSE', 'MAE', 'Errores de gradiente']
medias_valores_altos = [psnr_media, mse_media, mae_media, grad_diff_media]
desviaciones_valores_altos = [psnr_std, mse_std, mae_std, grad_diff_std]

# Métricas y datos para valores bajos
metricas_valores_bajos = ['SSIM', 'LPIPS', 'Correlacion de Pearson', 'Entropia cruzada', 'Diferencia de histograma']
medias_valores_bajos = [ssim_media, lpips_media, pearson_corr_media, cross_entropy_media, hist_diff_media]
desviaciones_valores_bajos = [ssim_std, lpips_std, pearson_corr_std, cross_entropy_std, hist_diff_std]

# Crear la figura con dos subgráficas
fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# Subgráfica para valores altos
axs[0].errorbar(metricas_valores_altos, medias_valores_altos, yerr=desviaciones_valores_altos, fmt='o', capsize=5, label='Media ± Desviación')
axs[0].set_xticklabels(metricas_valores_altos, rotation=45, ha='right')
axs[0].set_ylabel('Valor')
axs[0].set_title('Medias y Desviaciones Estándar de las Métricas con valores altos')
axs[0].legend()
axs[0].grid(True)

# Subgráfica para valores bajos
axs[1].errorbar(metricas_valores_bajos, medias_valores_bajos, yerr=desviaciones_valores_bajos, fmt='o', capsize=5, label='Media ± Desviación')
axs[1].set_xticklabels(metricas_valores_bajos, rotation=45, ha='right')
axs[1].set_ylabel('Valor')
axs[1].set_title('Medias y Desviaciones Estándar de las Métricas con valores bajos')
axs[1].legend()
axs[1].grid(True)

# Ajustar automáticamente el diseño para evitar superposiciones
plt.tight_layout()

plt.show()

# Lista de valores que quieres graficar
valores_grandes = [psnr_values, mse_values, mae_values, grad_diff_values]
valores_pequeños = [ssim_values, lpips_values, pearson_corr_values, cross_entropy_values, hist_diff_values]

# Lista de etiquetas para el histograma
etiquetas_pequeños = ['SSIM', 'LPIPS', 'Correlacion de Pearson', 'Entropia cruzada', 'Diferencia de histograma']
etiquetas_grandes = ['PSNR', 'MSE', 'MAE', 'Errores de gradiente']

# Crear figuras y ejes
fig, axs = plt.subplots(1, 2, figsize=(12, 8))

# Diagrama de caja valores pequeños (Boxplot)
axs[0].boxplot(valores_pequeños)
axs[0].set_xticklabels(etiquetas_pequeños, rotation=45, ha='right', fontsize=18)
axs[0].set_title('Boxplots de métricas con valores bajos', fontsize = 20)

# Diagrama de caja valores grandes (Boxplot)
axs[1].boxplot(valores_grandes)
axs[1].set_xticklabels(etiquetas_grandes, rotation=45, ha='right', fontsize=18)
axs[1].set_title('Boxplots de metricas con valores altos', fontsize = 20)

# Ajustar diseño de las subfiguras
plt.tight_layout()

# Mostrar los gráficos
plt.show()


# Crear subgráficos de histogramas para cada métrica
fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# Aplanar la matriz de subgráficos para iterar fácilmente sobre ella
axs = axs.flatten()

for i, (valor, etiqueta) in enumerate(zip(valores, metricas)):
    axs[i].hist(valor, bins=20, color='skyblue', edgecolor='black')
    axs[i].set_xlabel('Magnitud', fontsize = 14)
    axs[i].set_ylabel('Valor', fontsize = 14)
    axs[i].set_title(f'Histograma de {etiqueta}', fontsize = 14)
    axs[i].grid(True)

# Ajustar automáticamente el diseño para evitar superposiciones
plt.tight_layout()
plt.show()

## Final part, data representation of different experiments
# 1
# Crear subgráficos de histogramas para cada métrica de medias
fig_medias, axs_medias = plt.subplots(3, 3, figsize=(15, 15))
axs_medias = axs_medias.flatten()

metricas = ['PSNR', 'MSE', 'MAE', 'SSIM', 'LPIPS', 'Correlacion de Pearson', 'Entropia cruzada', 'Diferencia de histograma', 'Errores de gradiente']

for i, metrica in enumerate(metricas):
    for j in range(len(pruebas)):
        axs_medias[i].hist([medias_pruebas[j][i]], bins=10, alpha=0.5, label=pruebas[j])
    axs_medias[i].set_xlabel('Media')
    axs_medias[i].set_ylabel('Frecuencia')
    axs_medias[i].set_title(f'Histograma de medias de {metrica}')
    axs_medias[i].legend()

plt.tight_layout()
plt.show()

# Crear subgráficos de histogramas para cada métrica de coeficientes de variación
fig_cvs, axs_cvs = plt.subplots(3, 3, figsize=(15, 15))
axs_cvs = axs_cvs.flatten()

for i, metrica in enumerate(metricas):
    for j in range(len(pruebas)):
        axs_cvs[i].hist([CV_pruebas[j][i]], bins=10, alpha=0.5, label=pruebas[j])
    axs_cvs[i].set_xlabel('CV (%)')
    axs_cvs[i].set_ylabel('Frecuencia')
    axs_cvs[i].set_title(f'Histograma de CVs de {metrica}')
    axs_cvs[i].legend()

plt.tight_layout()
plt.show()

# 2
fig_medias, axs_medias = plt.subplots(3, 3, figsize=(15, 15))
axs_medias = axs_medias.flatten()

for i, metrica in enumerate(metricas):
    axs_medias[i].bar(pruebas, [medias_pruebas[j][i] for j in range(len(pruebas))], color='skyblue', edgecolor='black')
    axs_medias[i].set_xlabel('Pruebas')
    axs_medias[i].set_ylabel('Media')
    axs_medias[i].set_title(f'Medias de {metrica}')
    axs_medias[i].grid(True)

plt.tight_layout()
plt.show()

# Crear gráficos de barras para los coeficientes de variación
fig_cvs, axs_cvs = plt.subplots(3, 3, figsize=(15, 15))
axs_cvs = axs_cvs.flatten()

for i, metrica in enumerate(metricas):
    axs_cvs[i].bar(pruebas, [CV_pruebas[j][i] for j in range(len(pruebas))], color='skyblue', edgecolor='black')
    axs_cvs[i].set_xlabel('Pruebas')
    axs_cvs[i].set_ylabel('CV (%)')
    axs_cvs[i].set_title(f'CVs de {metrica}')
    axs_cvs[i].grid(True)

plt.tight_layout()
plt.show()

# 3
# Convertir a np.array para facilitar el manejo
medias_pruebas = np.array(medias_pruebas)
CV_pruebas = np.array(CV_pruebas)
# Crear subgráficos de boxplots para las medias
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()

for i, (ax, metrica) in enumerate(zip(axs, metricas)):
    ax.boxplot(medias_pruebas[:, i], patch_artist=True)
    ax.set_title(f'Boxplot de {metrica}')
    ax.set_xticks([1])  # Solo un conjunto de boxplots por métrica
    ax.set_xticklabels([metrica])

plt.tight_layout()
plt.show()

# Crear subgráficos de boxplots para los coeficientes de variación
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
axs = axs.flatten()

for i, (ax, metrica) in enumerate(zip(axs, metricas)):
    ax.boxplot(CV_pruebas[:, i], patch_artist=True)
    ax.set_title(f'Boxplot de CV de {metrica}')
    ax.set_xticks([1])  # Solo un conjunto de boxplots por métrica
    ax.set_xticklabels([metrica])

plt.tight_layout()
plt.show()

