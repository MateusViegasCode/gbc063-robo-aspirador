"""
Gera os gráficos dos resultados.

Uso:
    python graficos.py
Saída: resultados/curva_aprendizado.png e resultados/trajetoria.png
"""
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from stable_baselines3 import DQN

from ambiente import RoboAspiradorEnv, BASE, OBSTACULOS, SUJEIRAS, TAM

AZUL, LARANJA, CINZA, ESCURO = "#2457C5", "#C8601C", "#8A8F98", "#1B2330"
OTIMO = 7.0  # retorno da rota ótima (20 passos, 4 sujeiras, volta à base)

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E5E3DC", "grid.linewidth": 0.8,
                     "axes.edgecolor": "#8A8F98", "text.color": ESCURO, "axes.labelcolor": ESCURO})


def retorno_aleatorio(n=100):
    env, rng, rets = RoboAspiradorEnv(), np.random.default_rng(0), []
    for ep in range(n):
        env.reset(seed=ep)
        fim, tot = False, 0.0
        while not fim:
            _, r, t, tr, _ = env.step(rng.integers(5))
            tot += r
            fim = t or tr
        rets.append(tot)
    return np.mean(rets)


# ---------- 1. curva de aprendizado (avaliação gulosa a partir da base) ----------
fig, ax = plt.subplots(figsize=(9, 5))
variantes = [("sem", CINZA, "sem inícios exploratórios"), ("com", AZUL, "com inícios exploratórios")]
for chave, cor, rotulo in variantes:
    curvas = []
    for f in sorted(glob.glob(f"resultados/{chave}_inicio_aleatorio_seed*/evaluations.npz")):
        d = np.load(f)
        passos, ret = d["timesteps"], d["results"][:, 0]
        ax.plot(passos, ret, color=cor, lw=1, alpha=0.4)
        curvas.append(ret)
    if curvas:
        n = min(len(c) for c in curvas)
        media = np.mean([c[:n] for c in curvas], axis=0)
        ax.plot(passos[:n], media, color=cor, lw=2.5, label=f"{rotulo} (média de {len(curvas)} sementes)")

ax.axhline(OTIMO, color=ESCURO, ls="--", lw=1.2)
ax.text(5_000, OTIMO + 0.5, "rota ótima (7,0)", fontsize=12)
rand = retorno_aleatorio()
ax.axhline(rand, color=LARANJA, ls=":", lw=1.8)
ax.text(5_000, rand + 0.6, f"política aleatória ({rand:.1f})".replace(".", ","), fontsize=12, color=LARANJA)
ax.set_xlabel("passos de treino")
ax.set_ylabel("retorno do episódio a partir da base")
ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f} mil")
ax.legend(loc="lower right", frameon=False, fontsize=12)
fig.tight_layout()
fig.savefig("resultados/curva_aprendizado.png", dpi=200)

# ---------- 2. trajetória do melhor modelo ----------
env = RoboAspiradorEnv()
model = DQN.load("resultados/com_inicio_aleatorio_seed0/best_model")
obs, _ = env.reset()
caminho, aspirou, fim = [env.pos], [], False
while not fim:
    a, _ = model.predict(obs, deterministic=True)
    obs, r, t, tr, info = env.step(a)
    caminho.append(env.pos)
    if int(a) == 4 and r > 0:
        aspirou.append(env.pos)
    fim = t or tr

fig, ax = plt.subplots(figsize=(5, 5))
for x in range(TAM):
    for y in range(TAM):
        cor = "#3A4250" if (x, y) in OBSTACULOS else ("#D6E2F8" if (x, y) == BASE else "#FDFCF8")
        ax.add_patch(patches.Rectangle((x, y), 1, 1, facecolor=cor, edgecolor="#DDDAD0", lw=2))
for (x, y) in SUJEIRAS:
    ax.add_patch(patches.Circle((x + 0.5, y + 0.5), 0.2, color=LARANJA))
for i in range(len(caminho) - 1):
    (x0, y0), (x1, y1) = caminho[i], caminho[i + 1]
    if (x0, y0) != (x1, y1):
        ax.annotate("", xy=(x1 + 0.5, y1 + 0.5), xytext=(x0 + 0.5, y0 + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=AZUL, lw=2.5, mutation_scale=18))
for k, (x, y) in enumerate(aspirou, 1):  # ordem em que limpou
    ax.text(x + 0.5, y + 0.5, str(k), ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(0.2, 0.22, "B", ha="center", va="center", fontsize=14, fontweight="bold", color=AZUL)
ax.set_xlim(0, TAM)
ax.set_ylim(TAM, 0)  # (0,0) no canto superior esquerdo, igual ao Desafio 1
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig("resultados/trajetoria.png", dpi=200)
print(f"Trajetória: {env.passos} passos, sucesso={info['sucesso']}, ordem={aspirou}")
print("Gráficos salvos em resultados/")
