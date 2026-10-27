# Desafio 1 — Descrição do problema

Apresentação: [apresentacao.pdf](apresentacao.pdf)

## 1. Ambiente e estado markoviano

Sala em grade 5×5 com obstáculos fixos, uma base de recarga em (0,0) e 4 pontos de sujeira em posições fixas. O episódio acaba quando tudo está limpo e o robô volta à base, ou quando a bateria zera fora dela.

Estado: `s = (x, y, b, d1, d2, d3, d4)`

- `x, y ∈ {0,…,4}`: posição do robô
- `b ∈ {0,…,20}`: bateria
- `d1…d4 ∈ {0,1}`: se cada ponto de sujeira ainda está sujo

|S| = 25 × 21 × 2⁴ = 8.400 estados.

O estado é de Markov porque posição, bateria e mapa de sujeira bastam para prever o próximo estado dada a ação. Se o agente visse só a célula atual, precisaria lembrar o que já limpou (dependeria da história).

## 2. Ações

`A = {cima, baixo, esquerda, direita, aspirar}`

- Toda ação gasta 1 de bateria; bater em parede/obstáculo deixa o robô no lugar.
- Aspirar uma célula suja muda o `di` daquele ponto para 0; a sujeira não volta.
- Ao entrar na base a bateria volta para 20.
- Transições determinísticas, mas o agente não conhece o modelo.

## 3. Recompensas

| Evento | r |
|--------|---|
| Aspirar célula suja | +1 |
| Cada passo | −0,1 |
| Aspirar célula limpa ou bater na parede | −0,5 |
| Tudo limpo e de volta à base (fim do episódio) | +5 |
| Bateria zera fora da base (fim do episódio) | −10 |

Fator de desconto γ = 0,95.
