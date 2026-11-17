"""
Treina um DQN (Stable-Baselines3) no ambiente do robô aspirador.

Uso:
    python treinar_dqn.py                          # semente 0, com inícios exploratórios
    python treinar_dqn.py --seed 1                 # outra semente
    python treinar_dqn.py --sem-inicio-aleatorio   # treino sempre começando na base

A cada 5 mil passos a política gulosa é avaliada começando da base
(EvalCallback) e o melhor modelo é salvo em resultados/<nome>/best_model.zip.
"""
import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from ambiente import RoboAspiradorEnv

parser = argparse.ArgumentParser()
parser.add_argument("--passos", type=int, default=300_000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sem-inicio-aleatorio", action="store_true")
args = parser.parse_args()

inicio_aleatorio = not args.sem_inicio_aleatorio
nome = f"{'com' if inicio_aleatorio else 'sem'}_inicio_aleatorio_seed{args.seed}"
pasta = os.path.join("resultados", nome)
os.makedirs(pasta, exist_ok=True)

# ambiente de treino (o Monitor salva retorno e tamanho de cada episódio num CSV)
env = Monitor(RoboAspiradorEnv(inicio_aleatorio=inicio_aleatorio), os.path.join(pasta, "treino"),
              info_keywords=("sucesso", "limpos"))
# ambiente de avaliação: sempre começa na base com as 4 sujeiras
env_aval = Monitor(RoboAspiradorEnv())
aval = EvalCallback(env_aval, eval_freq=5_000, n_eval_episodes=1, deterministic=True,
                    best_model_save_path=pasta, log_path=pasta, verbose=0)

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,          # tamanho do replay buffer
    learning_starts=1_000,       # passos aleatórios antes de começar a treinar
    batch_size=64,
    gamma=0.95,                  # mesmo gama do Desafio 1
    train_freq=4,
    target_update_interval=500,  # rede alvo copiada a cada 500 passos
    exploration_fraction=0.3,    # ε cai de 1.0 até 0.05 nos primeiros 30% do treino
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[64, 64]),  # rede 7 -> 64 -> 64 -> 5
    seed=args.seed,
    verbose=0,
)

print(f"Treinando {nome} por {args.passos} passos...")
model.learn(total_timesteps=args.passos, callback=aval)
model.save(os.path.join(pasta, "final_model"))
print(f"Melhor retorno avaliado: {aval.best_mean_reward:.1f}  (modelos em {pasta}/)")
