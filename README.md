# INF721 - Deep Q-Learning Network
---
## Descrição

### O presente trabalho é corelacionado a parte inicial de uma inicião científica com o título: Aumentando a disponibilidade de recursos em redes móveis baseadas em computação em névoa utilizando aprendizado por reforço. Com o intuito de deslocar um agente por 3 ambiente fixos evitando ao máximo desconexões, uma arquitetura de Deep Q-Learning (DQN) foi proposta, relacionando Distâncias e SNR's (signal-to-noise ratio) de antenas próximas. 
---
## Pesos

### Os pesos de cada modelo treinado está disponível na pasta models, é possível identificar pelo nome qual ambiente está relacionado.

> Exemplo: DQN_wb-Env_80_0-0_45-60_p0.0002.pth

> DQN_wb-Env_Tamanho-ambiente_p1_p2_potencia.pth

### Onde p1 e p2 se referem respectivamente ao ponto de origem e destino do agente durante o treinamento, e potencia siginifica a potência de transmissão de cada antena colocada no ambiente.
---
## Tutorial

### Para realizar a inferencia dos modelos pré-treinado:


1.   baixe o repositório em sua máquina local.
2.   Abra um editor de código
3.   Execute o arquivo inf721_inference.py alterando nele qual ambiente gostaria de testar.
```
  $ python3 inf721_inference.py
```
4.   Para visualizar os testes, os arquivos .png ficam na pasta /plots com o nome no mesmo modelo dos pesos.


### Para relaizar o treinamento de algum modelo:

1.   baixe o repositório em sua máquina local.
2.   Abra um editor de código
3.   Execute o arquivo inf721_train.py alterando nele qual ambiente gostaria de testar, e os parâmetos disponíveis na apresentação: https://docs.google.com/presentation/d/1Y8gpoQe_vInr43thjJrwgsMKxyF3ao6HkJenPIQBfbU/edit?usp=sharing 
```
  $ python3 inf721_train.py
```
4.   Para visualizar os testes, os arquivos .png ficam na pasta /plots com o nome no mesmo modelo dos pesos.

Obs: Lembre-se de ter instalado as versões certas dos pacotes de importação.

---

