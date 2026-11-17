"""
Ambiente do robô aspirador (Desafio 1) no formato Gymnasium.

Grade 5x5, coordenadas (x, y) = (coluna, linha), com (0,0) no canto
superior esquerdo, igual ao desenho da apresentação do Desafio 1.

Estado: s = (x, y, b, d1, d2, d3, d4)
Ações:  0 = cima, 1 = baixo, 2 = esquerda, 3 = direita, 4 = aspirar
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

TAM = 5
BASE = (0, 0)
OBSTACULOS = {(2, 1), (2, 2)}
SUJEIRAS = [(4, 0), (0, 3), (3, 3), (4, 4)]
BATERIA_MAX = 20

# recompensas (tabela do Desafio 1)
R_PASSO = -0.1        # toda ação
R_LIMPAR = 1.0        # aspirar célula suja
R_INUTIL = -0.5       # aspirar célula limpa ou bater na parede/obstáculo
R_SUCESSO = 5.0       # tudo limpo e de volta à base
R_BATERIA = -10.0     # bateria zerou fora da base

MOVIMENTOS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
NOMES_ACOES = ["cima", "baixo", "esquerda", "direita", "aspirar"]


class RoboAspiradorEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, max_passos=100, inicio_aleatorio=False, render_mode=None):
        super().__init__()
        self.max_passos = max_passos  # truncamento (corte de segurança do treino)
        # inícios exploratórios: no treino o episódio pode começar numa posição
        # aleatória e com parte da sujeira já limpa (ajuda a explorar)
        self.inicio_aleatorio = inicio_aleatorio
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        # observação normalizada para a rede: x/4, y/4, b/20, d1..d4
        self.observation_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)

    def _obs(self):
        x, y = self.pos
        return np.array([x / (TAM - 1), y / (TAM - 1), self.bateria / BATERIA_MAX,
                         *self.sujo], dtype=np.float32)

    def estado(self):
        """Estado 'cru' (x, y, b, d1, d2, d3, d4), útil para imprimir."""
        return (*self.pos, self.bateria, *self.sujo)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = BASE
        self.bateria = BATERIA_MAX
        self.sujo = [1, 1, 1, 1]
        if self.inicio_aleatorio and self.np_random.random() < 0.5:
            livres = [(x, y) for x in range(TAM) for y in range(TAM) if (x, y) not in OBSTACULOS]
            self.pos = livres[self.np_random.integers(len(livres))]
            self.sujo = [int(v) for v in self.np_random.integers(0, 2, size=4)]
            if sum(self.sujo) == 0:
                self.sujo[self.np_random.integers(4)] = 1
        self.passos = 0
        self.limpos = 0
        return self._obs(), {}

    def step(self, acao):
        acao = int(acao)
        r = R_PASSO
        self.passos += 1
        self.bateria -= 1  # toda ação gasta 1 de bateria

        if acao in MOVIMENTOS:
            dx, dy = MOVIMENTOS[acao]
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < TAM and 0 <= ny < TAM and (nx, ny) not in OBSTACULOS:
                self.pos = (nx, ny)
            else:
                r += R_INUTIL  # bateu na parede, fica no lugar
        else:  # aspirar
            if self.pos in SUJEIRAS and self.sujo[SUJEIRAS.index(self.pos)] == 1:
                self.sujo[SUJEIRAS.index(self.pos)] = 0
                self.limpos += 1
                r += R_LIMPAR
            else:
                r += R_INUTIL

        if self.pos == BASE:
            self.bateria = BATERIA_MAX  # entrar/ficar na base recarrega

        terminou = False
        sucesso = False
        if self.pos == BASE and sum(self.sujo) == 0:
            r += R_SUCESSO
            terminou = sucesso = True
        elif self.bateria <= 0:
            r += R_BATERIA
            terminou = True

        truncou = (not terminou) and self.passos >= self.max_passos
        info = {"sucesso": sucesso, "limpos": self.limpos}
        return self._obs(), r, terminou, truncou, info

    def render(self):
        linhas = []
        for y in range(TAM):
            linha = ""
            for x in range(TAM):
                if (x, y) == self.pos:
                    c = "R"
                elif (x, y) == BASE:
                    c = "B"
                elif (x, y) in OBSTACULOS:
                    c = "#"
                elif (x, y) in SUJEIRAS and self.sujo[SUJEIRAS.index((x, y))]:
                    c = "*"
                else:
                    c = "."
                linha += c + " "
            linhas.append(linha)
        linhas.append(f"bateria={self.bateria:2d}  sujeiras restantes={sum(self.sujo)}")
        return "\n".join(linhas)
