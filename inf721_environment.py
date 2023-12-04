import numpy as np

class Antena:
    def __init__(self, x, y, p_trans=0.000003):
        self.x = x
        self.y = y

        self.wavelength = 0.125 # comprimento de onda
        self.sysloss = 1
        self.pi = np.pi
        self.min_dist = 0.25 # distancia minima antena

        self.p_trans = p_trans


    def calcula_snr(self, distancia):

        p_recW = self.p_trans*(self.wavelength / (4*self.pi*self.min_dist*self.sysloss))**2 # em Wats
        p_recDB = 10*np.log10(p_recW/0.001) # em decibel miliwatt

        if distancia < 1.0:
            distancia = 1.0

        snr = p_recDB + 20*np.log10(self.min_dist/distancia)

        return snr
    
class Environment:
    def __init__(self, tamanho_ambiente, ponto_origem, ponto_destino, p_trans=0.000003):

        self.tamanho_ambiente = tamanho_ambiente
        self.ponto_origem = ponto_origem
        self.ponto_destino = ponto_destino
        self.agent_pos = ponto_origem
        
        # Antenas definadas nas bordas
        self.antenas = [
                        Antena(0, 0, p_trans=p_trans),
                        Antena(0, tamanho_ambiente, p_trans=p_trans),
                        Antena(tamanho_ambiente, 0, p_trans=p_trans),
                        Antena(tamanho_ambiente,tamanho_ambiente, p_trans=p_trans)
                        ]

        self.conectado = self._verifica_conex(ponto_origem)

        self.name = self._create_name(p_trans)

    def reset(self):
        self.agent_pos = self.ponto_origem
        self.conectado = self._verifica_conex(self.ponto_origem)
        return self.get_state()

    def get_state(self):
        estado = []

        for antena in self.antenas:
            distancia = self._dist_manhattan([antena.x, antena.y], self.agent_pos)
            snr = antena.calcula_snr(distancia)
            estado.append(snr)

        distancia_destino = self._dist_manhattan(self.agent_pos, self.ponto_destino)

        estado.append(distancia_destino)

        return np.array(estado, dtype=np.float32)

    def step(self, acao):
        if acao == 0: # cima
            new_pos = (self.agent_pos[0], min(self.agent_pos[1] + 1, self.tamanho_ambiente ))
        elif acao == 1:# baixo
            new_pos = (self.agent_pos[0], max(self.agent_pos[1] - 1, 0))
        elif acao == 2: # esquerda
            new_pos = (max(self.agent_pos[0] - 1, 0), self.agent_pos[1])
        elif acao == 3: # direita
            new_pos = (min(self.agent_pos[0] + 1, self.tamanho_ambiente ), self.agent_pos[1])
        else:
            raise ValueError("Ação inválida!")

        recompensa = self._calcular_recompensa(new_pos)
        self.agent_pos = new_pos

        terminado = self.agent_pos == self.ponto_destino

        return self.get_state(), recompensa, terminado

    # DEBUG
    def get_agentPos(self):
        return self.agent_pos

    def get_dstPos(self):
        return self.ponto_destino

    def get_antennas(self):
        pos = []
        for antena in self.antenas:
            pos.append((antena.x,antena.y))
        return pos

    def is_conectado(self):
        return self.conectado

    def explorar_ambiente(self):
        historico_conexao = []  # Lista para armazenar o estado de conexão ao longo da exploração
        posicoes_agentes = []  # Lista para armazenar as posições do agente

        for x in range(self.tamanho_ambiente+1):
            for y in range(self.tamanho_ambiente+1):
                posicao_atual = (x, y)
                self.agent_pos = posicao_atual
                self.conectado = self._verifica_conex(posicao_atual)
                historico_conexao.append(self.conectado)
                posicoes_agentes.append(posicao_atual)

        # Restaurar a posição e o estado de conexão original após a exploração
        self.reset()

        return posicoes_agentes, historico_conexao

    ## PRIVATES
    def _create_name(self, p_trans):
        tam = str(self.tamanho_ambiente)
        src = str(self.ponto_origem[0]) + '-' + str(self.ponto_origem[1])
        trg = str(self.ponto_destino[0]) + '-' + str(self.ponto_destino[1])

        return '-Env_' + tam + '_' + src + '_' + trg + '_p' + str(p_trans)


    def _get_new_state(self, new_pos):
        estado = []

        for antena in self.antenas:
            distancia = self._dist_manhattan([antena.x, antena.y], new_pos)
            snr = antena.calcula_snr(distancia)
            estado.append(snr)

        distancia_destino = self._dist_manhattan(new_pos, self.ponto_destino)

        estado.append(distancia_destino)

        return np.array(estado, dtype=np.float32)

    def _verifica_conex(self, new_pos):
        snrs = np.asarray(self._get_new_state(new_pos)[:-1])
        snrs = snrs > -82 # threshold desconexcao

        if snrs.any():
            return True

        return False

    def _dist_manhattan(self, p1, p2):
        return np.sum(np.abs(np.array(p2) - np.array(p1)))

    def _calcular_recompensa(self, new_pos):

        recompensa_distancia = 0
        recompensa_antenas = 0

        distancia_atual_destino = self._dist_manhattan(self.agent_pos, self.ponto_destino)
        nova_distancia_destino = self._dist_manhattan(new_pos, self.ponto_destino)


        if nova_distancia_destino < distancia_atual_destino:
            recompensa_distancia -= 1
        else:
            recompensa_distancia -= 4


        conex = self._verifica_conex(new_pos)
        if self.conectado and not(conex):
            recompensa_antenas = -10
            self.conectado = False

        elif not(self.conectado) and conex:
            recompensa_antenas = -1
            self.conectado = True

        else:
          if self.conectado:
            recompensa_antenas = -0

          else:
            recompensa_antenas = -4


        # Combinando as recompensas com um pesos
        peso_distancia = 0.4  # Peso para a recompensa de distância
        recompensa_final = peso_distancia * recompensa_distancia + (1 - peso_distancia) * recompensa_antenas

        return recompensa_final
    
