"""
Avalia os modelos treinados começando da base, com as 4 sujeiras.

Uso:
    python avaliar.py              # tabela com todos os treinos em resultados/
    python avaliar.py --mostrar    # mostra um episódio passo a passo no terminal
    python avaliar.py --mostrar --modelo resultados/sem_inicio_aleatorio_seed0/best_model
"""
import argparse
import glob
import os
import time

import numpy as np
from stable_baselines3 import DQN

from ambiente import RoboAspiradorEnv, NOMES_ACOES

parser = argparse.ArgumentParser()
parser.add_argument("--modelo", default="resultados/com_inicio_aleatorio_seed0/best_model")
parser.add_argument("--mostrar", action="store_true")
args = parser.parse_args()


def episodio(env, politica, seed=None, mostrar=False):
    obs, _ = env.reset(seed=seed)
    if mostrar:
        print(env.render(), "\n")
    total, fim = 0.0, False
    while not fim:
        a = politica(obs)
        obs, r, term, trunc, info = env.step(a)
        total += r
        fim = term or trunc
        if mostrar:
            print(f"passo {env.passos:2d}: {NOMES_ACOES[int(a)]:8s}  r={r:+.1f}")
            print(env.render(), "\n")
            time.sleep(0.3)
    return total, info["sucesso"], info["limpos"], env.passos


env = RoboAspiradorEnv()

if args.mostrar:
    model = DQN.load(args.modelo)
    ret, suc, lim, pas = episodio(env, lambda o: model.predict(o, deterministic=True)[0], mostrar=True)
    print(f"Retorno total: {ret:.1f} | sujeiras limpas: {lim} | sucesso: {suc}")
else:
    # política aleatória: média de 100 episódios
    rng = np.random.default_rng(0)
    res = np.array([episodio(env, lambda o: rng.integers(5), seed=i) for i in range(100)], dtype=float)
    print(f"{'Política':36s} {'retorno':>8s} {'limpos':>7s} {'passos':>7s} {'sucesso':>8s}")
    print(f"{'Aleatória (média de 100 episódios)':36s} {res[:,0].mean():8.1f} {res[:,2].mean():7.1f} "
          f"{res[:,3].mean():7.1f} {100*res[:,1].mean():7.0f}%")
    # DQN: o ambiente e a política gulosa são determinísticos, então 1 episódio basta
    for pasta in sorted(glob.glob("resultados/*_seed*")):
        model = DQN.load(os.path.join(pasta, "best_model"))
        ret, suc, lim, pas = episodio(env, lambda o: model.predict(o, deterministic=True)[0])
        print(f"{'DQN ' + os.path.basename(pasta):36s} {ret:8.1f} {lim:7d} {pas:7d} {'sim' if suc else 'não':>8s}")
