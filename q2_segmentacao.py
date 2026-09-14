from skimage import data
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops




img = data.coins()

print(type(img)) #percebi que a classe da nossa imagem eh uma array
print(img.shape) #imprime as dimensoes da nossa matriz (no caso da imagem das coins eh 303 linhas por 384 colunas)
print(img.dtype) #mostra o tipo de cada numero armazenado na matriz, no caso unit8 (unsigned integer de 8 bits, ou seja, os numeros vao de 0 a 255)
print(img.min(), img.max()) #imprime o minimo e maximo da imagem, vamos usar isso pra segmentacao 

#plt.imshow(img, cmap='gray') #aqui estamos apenas preparando a imagem e dizendo que ela interprete cada numero como um tom de cinza
#plt.title("Imagem original")
#plt.show() #aqui mostra a janela com a imagem 






#-------------------------------------------#
#aqui vou aplicar a limiarizacao de otsu

limiar = threshold_otsu(img) #aplica a funcao do metodo otsu na imagem, guardando esse valor na variavel limiar
print("Limiar escolhido pelo otsu: ", limiar)

binaria = img > limiar #compara CADA numero de CADA pixel da imagem com o limiar calculado e armazena numa matriz do mesmo tamanho (binaria) que diz se o valor daquele pixel eh maior (true) ou menor (false) que o limiar

#plt.imshow(binaria, cmap='gray')
#plt.title(f"Binzarizada (limiar={limiar})")
#plt.show()

#plt.hist(img.ravel(), bins=256) #plotar o histograma para analisar melhor e entender pq esse limiar foi o escolhido pelo metodo otsu
#plt.show()





#------------------------------------------#
#aplicando agora operacoes morfologicas para resolver os problemas provenientes do metodo otsu, na qual vi que o canto superior, principalmente esquerdo, foi colocado como objeto uma parte que era fundo, e nas moedas exitem muitos pontos sendo classificados como fundo, mesmo sendo objetos. 


limpa = closing(binaria, disk(3))
limpa = opening(limpa, disk(3))

#plt.imshow(limpa, cmap='gray')
#plt.title("Depois da limpeza morfológica")
#plt.show()






#----------------------------------------#
#aplicando watershed



distancia = ndi.distance_transform_edt(limpa)

coords = peak_local_max(distancia, min_distance=20, labels=limpa) #essa funcao basicamente vai pegar todos os pontos de pico e armazenar numa matriz (numero de picos, 2), por se tratar de coordenadas, vamos ter a linha e a coluna, entao por exemplo: 5 picos, vamos ter 5 pontos diferentes na nossa imagem em (linhas, colunas)
mascara_picos = np.zeros(distancia.shape, dtype=bool) #criamos uma tela me branco para marcarmos somente os pontos de pico
#o numpy espera que, pra marcar varios pontos de picos de um vez, ele receba uma lista separada contedo somente os indices de linha e outra com os indices de ponto
mascara_picos[tuple(coords.T)] = True #fazemos a transposicao da matriz, e depois a ocmvertemos tambem numa tupla de dois arrays, sendo um de linhas e um de colunas. Assim o numpy vai enxergar diversos pontos discretos de uma vez e marcamos como true todos esses pontos (naquela tela de antes)
marcadores, _ = ndi.label(mascara_picos) #damos uma identidade para cada grupo de pixels true que esta interligado entre si

rotulos = watershed(-distancia, marcadores, mask=limpa)

n_moedas = rotulos.max()
print("Número de moedas detectadas:", n_moedas)

plt.imshow(rotulos, cmap='nipy_spectral')
plt.title(f"Segmentação final: {n_moedas} moedas")
plt.show()

#o trem dificil que foi de entender essa parte viu nossa senhora





#--------------------------------------------------------------------------------------------------------------------------#
#ainda assim, temos uma mancha roxa grandona no canto superior esquerdo, isso significa que aquela mancha que vimos la quando binzarizei sobreviveu tanto as opreações morfológicas quanto ao watershed, portanto, vamos utilizar outra alternativa para trata-las:

propriedades = regionprops(rotulos)
for regiao in propriedades: 
    print(f"Rótulo {regiao.label}: área = {regiao.area} pixels")

#no olhometro mesmo, podemos estabelecer esses limites aqui de areas minimas e maximas (colocando uma folga nas areas analisadas) pra se basear no restante
area_minima = 500
area_maxima = 5000

rotulos_filtrados = rotulos.copy()
n_validas = 0
for regiao in propriedades:
    if regiao.area < area_minima or regiao.area > area_maxima:
        rotulos_filtrados[rotulos_filtrados == regiao.label] = 0
    else:
        n_validas += 1

print("Número de moedas após filtro:", n_validas)

plt.imshow(rotulos_filtrados, cmap='nipy_spectral')
plt.title(f"Segmentação filtrada: {n_validas} moedas")
plt.show()
plt.imsave("saida_q2_segmentada.png", rotulos_filtrados, cmap='nipy_spectral')



#teste da iluminacao 
img_escura = (img * 0.5).astype('uint8')

limiar_escuro = threshold_otsu(img_escura)
print("Limiar na imagem escurecida:", limiar_escuro, "(original era", limiar, ")")

binaria_escura = img_escura > limiar_escuro
plt.imshow(binaria_escura, cmap='gray')
plt.title(f"Escurecida e binarizada (limiar={limiar_escuro})")
plt.show()



