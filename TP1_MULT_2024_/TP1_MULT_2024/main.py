#!/usr/bin/env python3
# -- coding: utf-8 --
"""Created on Fri Feb  9 14:59:30 2024
"""

import matplotlib.pyplot as plt
import matplotlib.colors as crl
import numpy
import numpy as np
import cv2
from PIL import Image
from scipy import fftpack

#variáveis globais
matrix = np.matrix([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
Q_Y = np.array([[16,  11,  10,  16,  24,  40,  51,  61],
               [12,  12,  14,  19,  26,  58,  60,  55],
               [14,  13,  16,  24,  40,  57,  69,  56],
               [14,  17,  22,  29,  51,  87,  80,  62],
               [18,  22,  37,  56,  68, 109, 103,  77],
               [24,  35,  55,  64,  81, 104, 113,  92],
               [49,  64,  78,  87, 103, 121, 120, 101],
               [72,  92,  95,  98, 112, 100, 103,  99]])


Q_CbCr= np.array([[17, 18, 24, 47, 99, 99, 99, 99],
               [18, 21, 26, 66, 99, 99, 99, 99],
               [24, 26, 56, 99, 99, 99, 99, 99],
               [47, 66, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99],
               [99, 99, 99, 99, 99, 99, 99, 99]])

# 3. Função para separar os canais RGB da imagem
def splitRGB(img):
    # Extração dos canais R, G e B da imagem
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B


# 3.5. Função para juntar os canais RGB em uma imagem
def joinRGB(R, G, B):
    # Criando uma imagem RGB a partir dos canais R, G e B
    imgRec = np.zeros((R.shape[0], R.shape[1], 3), dtype=np.uint8)
    imgRec[:, :, 0] = R
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B
    return imgRec


# 5.1. Função de conversão RGB para YCbCr
def RGB_to_YCbCr(R, G, B):
    # Matriz de conversão RGB para YCbCr
    #matrix = np.matrix([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]])
    # Calculando os canais Y, Cb e Cr
    Y = matrix[0, 0] * R + matrix[0, 1] * G + matrix[0, 2] * B
    Cb = (matrix[1, 0] * R + matrix[1, 1] * G + matrix[1, 2] * B) + 128
    Cr = (matrix[2, 0] * R + matrix[2, 1] * G + matrix[2, 2] * B) + 128
    return Y, Cb, Cr


# 5.2. Função inversa
def YCbCr_to_RGB(Y, Cb, Cr):
    # Matriz inversa de conversão YCbCr para RGB
    inverted = np.linalg.inv(matrix)
    # Calculando os canais R, G e B
    R = Y + inverted[0, 2] * (Cr - 128)
    G = Y + inverted[1, 1] * (Cb - 128) + inverted[1, 2] * (Cr - 128)
    B = Y + inverted[2, 1] * (Cb - 128) + inverted[2, 2] * (Cr - 128)
    # Garantir que os valores RGB estejam no intervalo [0, 255]
    R = np.round(R)
    G = np.round(G)
    B = np.round(B)
    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)
    return R, G, B


# 4.1. Função para aplicar padding aos canais RGB
def padRGB(img, h, w):
    #h -> número de linhas
    #w -> número de colunas
    # Calcular o preenchimento necessário para as linhas
    pad_height = 32 - (h % 32)
    pad_width = 32 - (w % 32)
    ## Repetir a primeira e última linha para preenchimento vertical
    lastLine = img[h-1, :][np.newaxis, :]
    bottom_pad = lastLine.repeat (pad_height, axis = 0)
    imagePadding = np.vstack([img, bottom_pad])
    lastColumn = imagePadding[:, w-1][: , np.newaxis]
    right_pad = lastColumn.repeat(pad_width, axis = 1)
    imagePadding = np.hstack([imagePadding, right_pad])

    return imagePadding


# 4.2. Função para remover padding dos canais RGB
def removePadding(img, h, w):
    # Obtendo as dimensões originais da imagem
    unpadding= img[0:h,0:w,::]
    return unpadding


# Função de codificação: separa os canais RGB e aplica padding

# Função para exibir imagem
def ShowImg(img, cmap=None, caption=" "):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.title(caption)
    plt.show()


# 5.3.1. Função para converter os canais RGB para YCbCr e visualizar cada canal
def RGB_to_YCbCr_and_visualize(R, G, B):
    # Converte os canais RGB para YCbCr
    Y, Cb, Cr = RGB_to_YCbCr(R, G, B)
    return Y, Cb, Cr


# Função para recuperar os canais RGB a partir dos canais YCbCr
def YCbCr_to_RGB_and_test(Y, Cb, Cr):
    # Converte os canais YCbCr para RGB
    R, G, B = YCbCr_to_RGB(Y, Cb, Cr)
    # Testa a recuperação dos valores originais de RGB com o pixel de coordenada [0, 0]
    print("Valores originais de RGB (Pixel [0, 0]):", R[0, 0], G[0, 0], B[0, 0])
    return R, G, B
#pergunta 6, downsamplig e upsampling
def downsampling (Y,Cb, Cr, ratio, flag):
    Cb_ratio = ratio[1]/ratio[0]
    if (ratio[2] == 4): # se o último elemento for 4, não há downsampling
        return Cb, Cr
    if ratio[2] == 0:
        if ratio[1] == 4:
            Cr_ratio = 0.5
        else:
            Cr_ratio = Cb_ratio
    else:
        Cr_ratio = 1

    imagemCbInterpolada = cv2.resize(Cb, None, fx=Cb_ratio, fy=Cr_ratio, interpolation=cv2.INTER_LINEAR)
    imagemCrInterpolada = cv2.resize(Cr, None, fx=Cb_ratio, fy=Cr_ratio, interpolation=cv2.INTER_LINEAR)
    if flag:
        ShowImg(Y, 'gray', "Downsampling Y")
        ShowImg(imagemCbInterpolada,'gray',"Downsampling Cb")
        ShowImg(imagemCrInterpolada, 'gray', "Downsampling Cr")
    return imagemCbInterpolada, imagemCrInterpolada

def upsampling (DsCb, DsCr, ratio: tuple, flag):
    Cb_ratio = ratio[0] / ratio[1]
    if (ratio[2] == 0):
        if (ratio[1] == 4):
            Cr_ratio = 0.5
        else:
            Cr_ratio = Cb_ratio
    else:
        Cr_ratio = 1
    imagemUpCbInterpolada = cv2.resize(DsCb, None, fx=Cb_ratio, fy=Cr_ratio, interpolation=cv2.INTER_LINEAR)
    imagemUpCrInterpolada = cv2.resize(DsCr, None, fx=Cb_ratio, fy=Cr_ratio, interpolation=cv2.INTER_LINEAR)
    if imagemUpCbInterpolada.shape !=  imagemUpCrInterpolada.shape:
        dimensao_minima = np.min(imagemUpCbInterpolada.shape[0], imagemUpCrInterpolada[1])
        imagemUpCbInterpolada = imagemUpCbInterpolada[:dimensao_minima, :dimensao_minima]
        imagemUpCrInterpolada = imagemUpCrInterpolada[:dimensao_minima, :dimensao_minima]
    if flag:
        ShowImg(imagemUpCbInterpolada, 'gray', "Upsampling Cb ")
        ShowImg(imagemUpCrInterpolada, 'gray', "Upampling Cr")
    return imagemUpCbInterpolada, imagemUpCrInterpolada


#Exercicio 7

#por serem matrizes
def DCT(channel):
    return fftpack.dct(fftpack.dct(channel, norm='ortho').T, norm='ortho').T
#7.2
def idct(channel):
    return fftpack.idct(fftpack.idct(channel, norm="ortho").T, norm="ortho").T

def imgDct(y, cb, cr, flag):
    YD = DCT(y)
    CbD= DCT(cb)
    CrD = DCT(cr)
    Y_dct = np.log(np.abs(YD) + 0.0001)
    Cb_dct = np.log(np.abs(CbD) + 0.0001)
    Cr_dct = np.log(np.abs(CrD) + 0.0001)
    if flag:
        ShowImg(Y_dct, 'gray', "Y DCT")
        ShowImg(Cb_dct, 'gray', "Cb DCT")
        ShowImg(Cr_dct, 'gray', "Cr DCT")
    return YD, CbD, CrD


def imgBlockDct(y, cb, cr, block, flag):
    Ydct = BlockDct(y, size=block)
    Cbdct = BlockDct(cb, size=block)
    Crdct = BlockDct(cr, size=block)
    Yblock = np.log(np.abs(Ydct) + 0.0001)
    Cbblock = np.log(np.abs(Cbdct) + 0.0001)
    Crblock = np.log(np.abs(Crdct) + 0.0001)
    if flag:
        ShowImg(Yblock, 'gray', "Y BLOCK DCT")
        ShowImg(Cbblock, 'gray', "Cb BLOCK DCT")
        ShowImg(Crblock, 'gray', "Cr BLOCK DCT")

    return Ydct, Cbdct, Crdct

def BlockDct(x, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = DCT(x[i:i+size, j:j+size])
    return newImg


def idct_block(y_dct, cb_dct, cr_dct, block, flag):

    y_inv = blockIdct(y_dct, size=block)
    cb_inv = blockIdct(cb_dct, size=block)
    cr_inv = blockIdct(cr_dct, size=block)

    Yshow = np.log(np.abs(y_inv) + 0.0001)
    Cbshow = np.log(np.abs(cb_inv) + 0.0001)
    Crshow = np.log(np.abs(cr_inv) + 0.0001)
    if flag:
        ShowImg(Yshow, 'gray', "IDCT Block Y")
        ShowImg(Cbshow, 'gray', "IDCT Block Cb")
        ShowImg(Crshow, 'gray', "IDCT Block Cr")
    return y_inv, cb_inv, cr_inv


def blockIdct(x: np.ndarray, size):
    h, w = x.shape
    newImg = np.zeros(x.shape)
    for i in range(0, h, size):
        for j in range(0, w, size):
            newImg[i:i+size, j:j+size] = idct(x[i:i+size, j:j+size])
    return newImg

def quantization(ycbcr, quality, flag):
    y, cb, cr = ycbcr
    factor_escala = (100 - quality) / 50 if quality >= 50 else 50 / quality
    QsY = np.round(Q_Y * factor_escala)
    QsC = np.round(Q_CbCr * factor_escala)
    QsY = np.clip(QsY, 1, 255).astype(np.uint8)
    QsC = np.clip(QsC, 1, 255).astype(np.uint8)
    if quality == 100:
        QsY = np.clip(QsY, 1, 1).astype(np.uint8)
        QsC = np.clip(QsC, 1, 1).astype(np.uint8)
    qy = np.empty(y.shape, dtype=y.dtype)
    qcb = np.empty(cb.shape, dtype=cb.dtype)
    qcr = np.empty(cr.shape, dtype=cr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            qy[i:i + 8, j:j + 8] = y[i:i + 8, j:j + 8] / QsY
    qy = np.round(qy).astype(np.int16)

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            qcb[i:i + 8, j:j + 8] = cb[i:i + 8, j:j + 8] / QsC
    qcb = np.round(qcb).astype(np.int16)

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            qcr[i:i + 8, j:j + 8] = cr[i:i + 8, j:j + 8] / QsC
    qcr = np.round(qcr).astype(np.int16)
    ly = np.log(np.abs(qy) + 0.0001)
    lcb = np.log(np.abs(qcb) + 0.0001)
    lcr = np.log(np.abs(qcr) + 0.0001)
    if flag:
        ShowImg(ly, 'gray', "Quantized Y")
        ShowImg(lcb, 'gray', "Quantized Cb")
        ShowImg(lcr, 'gray', "Quantized Cr")
    return qy, qcb, qcr

def IQuantization(ycbcr, quality, flag):
    qy, qcb, qcr = ycbcr
    factor_escala = (100 - quality) / 50 if quality >= 50 else 50 / quality
    QsY = np.round(Q_Y * factor_escala)
    QsC = np.round(Q_CbCr * factor_escala)
    QsY = np.clip(QsY, 1, 255).astype(np.uint8)
    QsC = np.clip(QsC, 1, 255).astype(np.uint8)
    if quality == 100:
        QsY = np.clip(QsY, 1, 1).astype(np.uint8)
        QsC = np.clip(QsC, 1, 1).astype(np.uint8)
    y = np.empty(qy.shape, dtype=qy.dtype)
    cb = np.empty(qcb.shape, dtype=qcb.dtype)
    cr = np.empty(qcr.shape, dtype=qcr.dtype)

    for i in range(0, y.shape[0], 8):
        for j in range(0, y.shape[1], 8):
            y[i:i+8, j:j+8] = qy[i:i+8, j:j+8] * QsY

    for i in range(0, cb.shape[0], 8):
        for j in range(0, cb.shape[1], 8):
            cb[i:i+8, j:j+8] = qcb[i:i+8, j:j+8] * QsC

    for i in range(0, cr.shape[0], 8):
        for j in range(0, cr.shape[1], 8):
            cr[i:i+8, j:j+8] = qcr[i:i+8, j:j+8] * QsC

    ly = np.log(np.abs(qy) + 0.0001)
    lcb = np.log(np.abs(qcb) + 0.0001)
    lcr = np.log(np.abs(qcr) + 0.0001)
    if flag:
        ShowImg(ly, 'gray',"iQuantized Y  Qualidade:" + str(quality))
        ShowImg(lcb, 'gray', "iQuantized Cb  Qualidade:" + str(quality))
        ShowImg(lcr, 'gray', "iQuantized Cr  Qualidade:" + str(quality))

    return y.astype(float), cb.astype(float), cr.astype(float)

def DPCM(imgDCTQuant, channel, flag):
    imgDPCM = imgDCTQuant.copy()
    dc0 = imgDPCM[0, 0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCTQuant[i, j]
            diff = dc - dc0
            dc0 = dc
            imgDPCM[i, j] = diff
    imgDpcm = np.log(np.abs(imgDPCM) + 0.0001)
    if flag:
        ShowImg(imgDpcm, 'gray', "DPCM ("+ str(channel) + ")")

    return imgDPCM

def iDPCM(imgDCT_Q, channel, flag):
    # DPCM 8x8
    imgDPCM = imgDCT_Q.copy()
    dc0 = imgDPCM[0, 0]
    nl, nc = imgDPCM.shape
    for i in range(0, nl, 8):
        for j in range(0, nc, 8):
            if i == 0 and j == 0:
                continue
            dc = imgDCT_Q[i, j]
            s = dc + dc0
            dc0 = s
            imgDPCM[i, j] = s

    imgshow = np.log(np.abs(imgDPCM) + 0.0001)
    if flag:
        ShowImg(imgshow, 'gray', "IDPCM (" + str(channel) + ")")

    return imgDPCM

#10.4-> metricas de distorção

def MSE(imgOriginal,imgReconbstruida,h, w):
    return np.sum((imgOriginal.astype(float) - imgReconbstruida.astype(float)) **2)/(h*w)

def RMSE (imgOriginal,imgReconbstruida,h, w):
    return numpy.sqrt(MSE(imgOriginal, imgReconbstruida, h, w))

def SNR (imgOriginal, imgReconstruida,h,w):
    potencia_sinal = np.sum((imgOriginal.astype(float)**2))/(h*w)
    m = MSE(imgOriginal, imgReconstruida, h,w)
    return 10*np.log10((potencia_sinal/m))

def PSNR(imgOriginal, imgReconstruida, h,w):
    m = MSE(imgOriginal, imgReconstruida, h, w)
    return 10*np.log10((np.max(imgOriginal)**2)/m)

def diference_image(imgOriginal, imgReconstruida, flag):
    YOr, CbOr, CrOr = RGB_to_YCbCr(imgOriginal[:, :, 0], imgOriginal[:, :, 1], imgOriginal[:, :, 2])
    YRec, CbRec, CrRec = RGB_to_YCbCr(imgReconstruida[:, :, 0], imgReconstruida[:, :, 1], imgReconstruida[:, :, 2])
    diferencas = np.absolute(YOr-YRec)
    if flag:
        ShowImg(diferencas, 'gray', "Imagem Diferenças")
    return diferencas

def max_diff(imgOriginal, imgReconstruida, flag):
    MaxDiff = diference_image(imgOriginal, imgReconstruida, flag)
    return np.max(MaxDiff)


def avg_diff(imgOriginal, imgReconstruida, flag):
    AvgDiff = diference_image(imgOriginal, imgReconstruida, flag)
    return numpy.average(AvgDiff)


def encoder(img, h, w, quality, flag):

    cm_red = crl.LinearSegmentedColormap.from_list("red", [(0, 0, 0), (1, 0, 0)], N=256)
    cm_green = crl.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256)
    cm_blue = crl.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)

    # Separando os canais RGB da imagem
    # Aplicando padding aos canais RGB
    img = padRGB(img, h, w)
    R, G, B = splitRGB(img)
    Y, Cb, Cr = RGB_to_YCbCr(R, G, B)
    DownCb, DownCr = downsampling(Y,Cb, Cr, (4,2,0), flag)
    print(f'Dimensões da imagem DownCb: {DownCb.shape}')
    print(f'Dimensões da imagem DownCr: {DownCr.shape}')
    if flag:
        ShowImg(R, cmap=cm_red, caption="Red com padding")
        ShowImg(G, cmap=cm_green, caption="Green com padding")
        ShowImg(B, cmap=cm_blue, caption="Blue com padding")
        ShowImg(Y, cmap='gray', caption="Y Channel")
        ShowImg(Cb, cmap='gray', caption="Cb Channel")
        ShowImg(Cr, cmap='gray', caption="Cr Channel")
    imgDct(Y, DownCb, DownCr, flag)
    Yblock, Cbblock, Crblock= imgBlockDct(Y,DownCb, DownCr, 8, flag)
    quantY, quantCb, quantCr = quantization((Yblock, Cbblock, Crblock), quality, flag)
    quantY = DPCM(quantY , "y", flag)
    quantCb = DPCM(quantCb, "cb", flag)
    quantCr = DPCM(quantCr, "cr", flag)

    return quantY, quantCb, quantCr


# Função de decodificação: remove padding dos canais RGB
def decoder(y_1, cb_1, cr_1, h, w, quality, flag):
    y_1= iDPCM(y_1, "y", flag)
    cb_1 = iDPCM(cb_1, "cb",flag)
    cr_1 = iDPCM(cr_1, "cr", flag)
    y_dct, cb_dct, cr_dct = IQuantization((y_1, cb_1, cr_1), quality, flag)
    y,cb,cr = idct_block(y_dct,cb_dct,cr_dct, 8, flag)
    cb, cr = upsampling(cb, cr, (4, 2, 0), flag)
    R, G, B = YCbCr_to_RGB(y, cb, cr)
    mat = np.zeros((h, w, 3), dtype=np.uint8)
    mat[:, :, 0] = R[:h, :w]
    mat[:, :, 1] = G[:h, :w]
    mat[:, :, 2] = B[:h, :w]

    return mat


# Função principal
def main():

    fname = "airport.bmp"
    img_bgr = cv2.imread(fname)
    print(img_bgr.shape)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    """
    imagem = Image.open(fname)
    imagem_rgb = imagem.convert('RGB')r,g,b = imagem_rgb.getpixel((0,0))
    print(f' ({r}),({g}),({b}) ' )"""
    ShowImg(img, cmap=None, caption="original")
    h = img.shape[0]
    w = img.shape[1]
    quality= 100
    y, cb, cr = encoder(img, h, w, quality, True)
    xpto = decoder(y,cb, cr, h, w, quality, True)
    diference_image(img, xpto, True)
    mse = MSE(img, xpto,h, w)
    rmse = RMSE(img, xpto, h, w)
    snr = SNR(img,xpto, h, w)
    psnr = PSNR(img, xpto,h,w)
    md = max_diff(img,xpto,False)
    ad = avg_diff(img, xpto, False)
    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("SNR: " + str(snr))
    print("PSNR: "+ str(psnr))
    print("MAX DIFF: " + str(md))
    print("AVG DIFF: " + str(ad))
    ShowImg(xpto, cmap=None, caption="Imagem reconstruida")

if __name__ == "__main__":
    main()

