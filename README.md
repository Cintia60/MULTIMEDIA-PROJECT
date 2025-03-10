# Trabalho Prático - Compressão de Imagem com Codec JPEG

Este repositório contém a implementação e análise de mecanismos utilizados na compressão de imagens através do codec JPEG, utilizando Python.

## Objetivo

O objetivo deste trabalho prático é proporcionar uma compreensão sobre as técnicas de compressão de imagem, com ênfase na compressão utilizando o codec JPEG. O trabalho abrange diversos tópicos, como:

- Compressão de imagens em formato BMP para JPEG com diferentes qualidades.
- Conversão entre modelos de cores RGB e YCbCr.
- Aplicação de transformadas, como a Transformada Discreta de Cosseno (DCT).
- Quantização e codificação de coeficientes.

## Estrutura do Projeto

Este projeto é composto por diversas etapas de processamento de imagens, que são descritas nas seções abaixo.

### 1. Compressão de Imagens BMP para JPEG
As imagens fornecidas no formato BMP são comprimidas para o formato JPEG utilizando diferentes qualidades de compressão (alta, média e baixa).

- **Qualidade alta (Q=75)**
- **Qualidade média (Q=50)**
- **Qualidade baixa (Q=25)**

A compressão é realizada utilizando um editor de imagem como o GIMP. Após a compressão, é feita uma comparação entre os resultados obtidos para diferentes qualidades.

### 2. Visualização de Imagem no Modelo RGB
O processo de visualização da imagem em RGB é realizado com as seguintes etapas:
- Leitura de uma imagem BMP.
- Separação da imagem nos componentes RGB.
- Visualização dos canais RGB individualmente.

### 3. Pré-processamento: Padding
Caso a dimensão da imagem não seja múltipla de 32x32, é feito um processo de padding, replicando a última linha e coluna da imagem.

### 4. Conversão para o Modelo de Cor YCbCr
A imagem é convertida do modelo de cor RGB para o modelo YCbCr, um passo fundamental no processo de compressão JPEG. A inversão da conversão também é realizada para garantir que a imagem original possa ser recuperada.

### 5. Sub-amostragem (Downsampling)
Os canais Y, Cb, e Cr da imagem são sub-amostrados de acordo com as especificações do codec JPEG (4:2:0, 4:2:2). A sub-amostragem é realizada utilizando a função `cv2.resize` com diferentes métodos de interpolação.

### 6. Transformada Discreta de Cosseno (DCT)
A DCT é aplicada aos canais da imagem (Y, Cb, Cr) tanto no formato completo quanto em blocos de 8x8 e 64x64. Essa etapa permite a compressão eficiente da imagem, preservando as características mais importantes.

### 7. Quantização
Os coeficientes da DCT são quantizados para reduzir a quantidade de dados necessários para representar a imagem. A quantização é realizada com diferentes fatores de qualidade, que impactam diretamente na taxa de compressão e na qualidade final da imagem.

### 8. Codificação DPCM dos Coeficientes DC
Os coeficientes DC são codificados utilizando a técnica de Diferença de Previsão de Código (DPCM), que reduz ainda mais a quantidade de dados necessários para representar a imagem.

### 9. Codificação e Descodificação End-to-End
Após a quantização e a codificação DPCM, o código realiza a compressão e descodificação da imagem de forma completa, com a reconstrução da imagem comprimida. A comparação entre a imagem original e a imagem descodificada é feita utilizando métricas de distorção (MSE, RMSE, PSNR, etc.).

## Como Executar

### Requisitos

- Python 3.x
- Bibliotecas necessárias:
  - `numpy`
  - `cv2` (OpenCV)
  - `scipy`
  - `matplotlib`

Você pode instalar as dependências utilizando `pip`:

```bash
pip install numpy opencv-python scipy matplotlib
