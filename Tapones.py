import cv2
import matplotlib.pyplot as plt
import numpy as np

def bwareaopen(img, min_size, connectivity=8):
        """Remove small objects from binary image (approximation of 
        bwareaopen in Matlab for 2D images).
    
        Args:
            img: a binary image (dtype=uint8) to remove small objects from
            min_size: minimum size (in pixels) for an object to remain in the image
            connectivity: Pixel connectivity; either 4 (connected via edges) or 8 (connected via edges and corners).
    
        Returns:
            the binary image with small objects removed
        """
    
        # Find all connected components (called here "labels")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            img, connectivity=connectivity)
        
        # check size of all connected components (area in pixels)
        for i in range(num_labels):
            label_size = stats[i, cv2.CC_STAT_AREA]
            
            # remove connected components smaller than min_size
            if label_size < min_size:
                img[labels == i] = 0
                
        return img

def rellenar_agujeros(img):
    alto,ancho = img.shape
    mask = np.zeros((alto+2, ancho+2), np.uint8)        # Mask used to flood filling.
    im_floodfill = img.copy()                         # Copy the thresholded image.
    cv2.floodFill(im_floodfill, mask, (0,0), 255);      # Floodfill from point (0, 0)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)    # Invert floodfilled image
    img = img | im_floodfill_inv                    # Combine the two images to get the foreground.
    return img

def distanciaColor(imagen,pixel,color):
    """ Devuelve la 'distancia de color' que existe entre
    el color de un píxel de la imagen y un color determinado.
        - imagen
        - pixel: tupla con los valores (j,i)
        - color: tupla con los valores (B,G,R)
    """
    exponente = 2
    colorPixel = imagen[pixel[1],pixel[0]]
    distancia = ((colorPixel[0]-color[0])**exponente+
                 (colorPixel[1]-color[1])**exponente+
                 (colorPixel[2]-color[2])**exponente)**(1/exponente)
    return distancia

# # PROBLEMAS:
# - Algunas sombras --> O cambiar la segmentación o limpiar pixeles de color cercano al de las sombras
# - Algunos bordes --> usar algo como clearborders
# - Limpiar algunos objetos pequeños

# Ruta de la imagen
numeroFoto = 2
ruta = "/home/siali/Github/FormacionSiali/Artificial-Vision/Proyecto_Tapones/final_project_imgs/" + str(numeroFoto) + ".jpg"

# Leer la imagen, pasar a RGB, copiar y sacar las dimensiones
imagen = cv2.imread(ruta)
imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
alto,ancho,_ = imagen.shape

# Reescalado de la imagen
imagen = cv2.resize(imagen,(round(ancho/4),round(alto/4)))
copia = imagen.copy()
alto,ancho,_ = copia.shape

# Imagen de bordes
bordes = cv2.Canny(copia,100,200)

# Dilatación de bordes
tam_mask = 3
mask_dil = np.ones((tam_mask,tam_mask),np.uint8)
bordes = cv2.dilate(bordes,mask_dil,iterations=2)
cv2.imshow("bordes",bordes)


# # Filtrado paso bajo de la imagen
# #   - Pruebo distintos tamaños de kernel
# for i in range(9):
#     tam = 15*(i+1)
#     kernel = np.ones((tam, tam),np.float32)/tam**2 # area to calculate the average
#     smoothed = cv2.filter2D(copia,-1, kernel)  # src, depth(-1 same as the src), area
#     plt.subplot(3,3,i+1)
#     plt.imshow(smoothed, cmap='gray')
#     plt.title("Kernel "+str(tam)+"x"+str(tam))
# plt.show()
tam = 10
kernel = np.ones((tam, tam),np.float32)/tam**2 # area to calculate the average
copia = cv2.filter2D(copia,-1, kernel)  # src, depth(-1 same as the src), area

# Comprobación de las matrices HSV para ver con cual segmentar
hsv = cv2.cvtColor(copia,cv2.COLOR_RGB2HSV)
h,s,v = cv2.split(hsv)

# # Representación de las matrices HSV
# fig, axes = plt.subplots(1, 3, figsize=(100, 125))
# axes[0].imshow(h, cmap='gray')
# axes[0].set_title('H')
# axes[1].imshow(s, cmap='gray')
# axes[1].set_title('S')
# axes[2].imshow(v, cmap='gray')
# axes[2].set_title('V')
# plt.show()

# Segmentación con la suma de los canales H y S
copia = cv2.bitwise_or(h,s)

# Representación para escoger un umbral correcto
# for i in range(9):
#     umbral = round(85+i*(125-85)/9)
#     _,foto = cv2.threshold(copia,umbral,255,cv2.THRESH_BINARY)
#     plt.subplot(3,3,i+1)
#     plt.imshow(foto, cmap='gray')
#     plt.title("Umbral: "+str(umbral))
# plt.show()
umbral = 100
_,copia = cv2.threshold(copia,umbral,255,cv2.THRESH_BINARY)

# Relleno de agujeros
copia = rellenar_agujeros(copia)

# Erosion
tam_mask = 3
mask_erosion = np.ones((tam_mask,tam_mask),np.uint8)
copia = cv2.erode(copia,mask_erosion,iterations=2)

notBordes = cv2.bitwise_not(bordes)

copia = cv2.bitwise_and(copia,notBordes)

copia = bwareaopen(copia,ancho*alto*.0005)

# cv2.imwrite("/home/siali/Github/FormacionSiali/Artificial-Vision/Proyecto_Tapones/final_project_imgs/Imagen"+str(numeroFoto)+".jpg",copia)

# ## Segmentar el fondo
# lim_inf = np.array([0,5,30])
# lim_sup = np.array([180,50,210])
# mask_fondo = cv2.inRange(hsv,lim_inf,lim_sup)
# fondo = cv2.bitwise_and(imagen,imagen,mask=mask_fondo)
# # copia = cv2.resize(copia,(round(ancho/4),round(alto/4)))
# cv2.imshow("fondo",cv2.bitwise_not(mask_fondo))
# cv2.imshow("hsv",hsv)

tapones = []
colores = {"Blanco": (255,255,255),"Negro": (0,0,0),"Azul": (255,0,0),"Rojo": (0,0,255),"Amarillo": (0,255,255)}
contours,hierarchy = cv2.findContours(copia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cont in contours:
    M = cv2.moments(cont)
    if M["m00"] != 0:
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        # cv2.circle(imagen,(cx,cy),3,(0,0,255),-1)
    area = cv2.contourArea(cont)

    # distancias = []
    # for i in colores:
    #      distancias.append(distanciaColor(imagen,(cx,cy),colores[i]))

    # Padreada máxima con list comprehension
    distancias = [distanciaColor(imagen,(cx,cy),colores[i]) for i in colores]

    colorCercano = min(distancias)
    indice = distancias.index(colorCercano)

    # Creo el diccionario tapon y represento los atributos de tapón
    tapon = {"area": area, "color": list(colores)[indice], "centro": (cx,cy)}
    cv2.putText(imagen,str(tapon["area"]),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255))
    cv2.putText(imagen,tapon["color"],(cx,cy+round(alto/100)),cv2.FONT_HERSHEY_COMPLEX,.5,(255,255,255))
    cv2.circle(imagen,tapon["centro"],3,(0,0,255),-1)

    
# copia = cv2.resize(copia,(round(ancho/4),round(alto/4)))
cv2.imshow("imagen",copia)
cv2.imshow("original",imagen)

cv2.waitKey(0)
cv2.destroyAllWindows()