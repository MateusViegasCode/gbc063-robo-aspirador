"""
Confere se a rota do DQN é a ótima, calculando a política ótima exata
por iteração de valor, usando o modelo do ambiente.

Uso:
    python verificar_otimo.py
"""
import itertools

from ambiente import (TAM, BASE, OBSTACULOS, SUJEIRAS, BATERIA_MAX, MOVIMENTOS,
                      R_PASSO, R_LIMPAR, R_INUTIL, R_SUCESSO, R_BATERIA)

GAMMA = 0.95


def transicao(s, a):
    """Mesma dinâmica do ambiente.step(). Devolve (próximo estado ou None se terminou, recompensa)."""
    x, y, b, sujo = s
    sujo = list(sujo)
    r = R_PASSO
    b -= 1
    pos = (x, y)
    if a in MOVIMENTOS:
        dx, dy = MOVIMENTOS[a]
        nova = (x + dx, y + dy)
        if 0 <= nova[0] < TAM and 0 <= nova[1] < TAM and nova not in OBSTACULOS:
            pos = nova
        else:
            r += R_INUTIL
    else:
        if pos in SUJEIRAS and sujo[SUJEIRAS.index(pos)] == 1:
            sujo[SUJEIRAS.index(pos)] = 0
            r += R_LIMPAR
        else:
            r += R_INUTIL
    if pos == BASE:
        b = BATERIA_MAX
    if pos == BASE and sum(sujo) == 0:
        return None, r + R_SUCESSO
    if b <= 0:
        return None, r + R_BATERIA
    return (pos[0], pos[1], b, tuple(sujo)), r


estados = [(x, y, b, d) for x in range(TAM) for y in range(TAM) if (x, y) not in OBSTACULOS
           for b in range(1, BATERIA_MAX + 1) for d in itertools.product([0, 1], repeat=4)]


def q(V, s, a):
    ns, r = transicao(s, a)
    return r + (0 if ns is None else GAMMA * V[ns])


# iteração de valor: V(s) <- max_a [ r + γ V(s') ]
V = {s: 0.0 for s in estados}
for it in range(1, 1000):
    delta = 0.0
    for s in estados:
        novo = max(q(V, s, a) for a in range(5))
        delta = max(delta, abs(novo - V[s]))
        V[s] = novo
    if delta < 1e-9:
        break

# segue a política ótima a partir da base com tudo sujo
s = (BASE[0], BASE[1], BATERIA_MAX, (1, 1, 1, 1))
passos, retorno, ordem = 0, 0.0, []
while s is not None:
    a = max(range(5), key=lambda a: q(V, s, a))
    ns, r = transicao(s, a)
    if a == 4 and r > 0:
        ordem.append((s[0], s[1]))
    retorno += r
    passos += 1
    s = ns

print(f"Iteração de valor convergiu em {it} iterações ({len(estados)} estados).")
print(f"Política ótima: {passos} ações, retorno {retorno:.1f}")
print("Ordem em que limpa:", " -> ".join(str(p) for p in ordem))
