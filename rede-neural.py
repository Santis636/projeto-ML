#usando keras inceptionV3 para analise de imagens
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

transform = transforms.ToTensor() # definindo a conversão da imagem para tensor
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform) # carrega a parte de treino no dataset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True) 

valset = datasets.MNIST('./MNIST_data/', download=True, train=False, transform=transform)  # carrega a parte de validação do dataset Use train=False para validação
valloader = DataLoader(valset, batch_size=64, shuffle=True) # cria um buffer para pegar os dados por partes

dataiter = iter(trainloader)
images, etiquetas  = next(dataiter)
plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')

print(images[0].shape)
print(etiquetas[0].shape)

class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__() 
        self.linear1 = nn.Linear(28*28,128) # camada de entrada, 784 neurônios que se ligam a 128 
        self.linear2 = nn.Linear(128,64) # camada interna 1, 128 neurônios que se ligam a 64 
        self.linear3 = nn.Linear(64,10) # camada interna 2, 64 neurônios que se ligam a 10
        # para camada de saida não é necessário definir nada pois só precisamos pegar o output da camada interna 2 
        
    def forward(self,X):
        X = F.relu(self.linear1(X)) # função de ativação da camada de entrada para camada interna 1
        X = F.relu(self.linear2(X)) # função de ativação da camada interna 1 para cama interna 2
        X = self.linear3(X) # função de ativação da camada interna 2 para a camada de saída, nesse caso f(x)=x
        return F.log_softmax(X, dim=1) # dados utilizados para calcular a perda
    
def treino(modelo, trainloader, device):
    otimizador = optim.SGD(modelo.parameters(), lr=0.01, momentum=0.5) # define a politica de atualização dos pesos e bais
    inicio = time() # timer para saber quanto tempo levou
    criterio = nn.NLLLoss() # definindo o criterio para calcular a perda
    EPOCHS = 10 # numeros de epochs que o algoritimo rodará
    modelo.train() # ativando o modo de treinamento do modelo 
    
    for epoch in range(EPOCHS):
        perda_acumulada= 0
        
        for imagens, etiquetas in trainloader:
            imagens = imagens.view(imagens.shape[0], -1) # converter as imagens para "vetores" de 28*28 casas para ficarem compativeis com a estrutura
            otimizador.zero_grad() # zerando os gradientes por conta do ciclo anterior
            
            output = modelo(imagens.to(device)) # colocando os dados no modelo
            perda_instantanea = criterio(output, etiquetas.to(device)) # calculando a perda da epoch em questão
            
            perda_instantanea.backward() # back progaation a partir da perda
            
            otimizador.step() # atualizando os pesos e bias
            
            perda_acumulada += perda_instantanea.item() # atualizaçaõ da perda acumulada
        else:
            print(f'Epoch{epoch+1} - Perda resultante: {(perda_acumulada/len(trainloader)):.4f}')
    print(f'\nTempo de treino (em minutos) = {((time()-inicio)/60):.2f}')

def validacao(modelo, valloader, device):
    conta_corretas, conta_todas = 0,0
    for imagens, etiquetas in valloader:
        for i in range(len(etiquetas)):
            img = imagens[i].view(1,784)
            # desativa o autograd para acelerar a validação. Grafos computacionais dinâmicos tem um alto custo de processamento
            with torch.no_grad():
                logps = modelo(img.to(device)) # output do modelo em escala logaritmica
                
            ps = torch.exp(logps) # converte output para escala normal(lembrando que é um tensor)
            probab = list(ps.cpu().numpy()[0])
            etiqueta_pred = probab.index(max(probab)) # converte o tensor em um número, no caso, um número que o modelo previu
            etiqueta_certa = etiquetas.numpy()[i]
            if (etiqueta_certa == etiqueta_pred): 
                conta_corretas += 1
            conta_todas += 1
    
    print(f'Total de imagens testadas = {conta_todas}')
    print(f'\nPrecisão do modelo = {conta_corretas*100/conta_todas}')

modelo = Modelo()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modelo.to(device)
treino(modelo, trainloader, device)
validacao(modelo, valloader, device)

#GPTCHAT - tests
# Determinando o dispositivo (CPU ou GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Dispositivo: {device}")

# # Criando uma instância do modelo e movendo-o para o dispositivo
# modelo = Modelo().to(device)

# # Treinando o modelo
# treino(modelo, trainloader, device)