import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_train(env_name, avg_scores, n_steps):
  fig, ax1 = plt.subplots(figsize=(10, 6))

  # Gráfico para avg_score
  color = 'tab:blue'
  ax1.set_xlabel('Época')
  ax1.set_ylabel('Recompensa média', color=color)
  ax1.plot(range(1, len(avg_scores) + 1), avg_scores, label='avg_score', color=color)
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.legend(loc='upper left')

  # Cria um segundo eixo y para n_steps
  ax2 = ax1.twinx()
  color = 'tab:green'
  ax2.set_ylabel('Número de Passos', color=color)
  ax2.plot(range(1, len(n_steps) + 1), n_steps, label='n_steps', color=color)
  ax2.tick_params(axis='y', labelcolor=color)
  ax2.legend(loc='upper right')

  # Título geral
  plt.title('Recompensa média e Número de Passos ao longo das Épocas')

  # Exibe o gráfico
  plt.savefig('plots/'+ env_name + '_train.png')

def choose_action(model, observation):
    state = torch.tensor(observation)
    actions = model.forward(state)
    action = torch.argmax(actions).item()

    return action

def inference_route(model, env):
    route = []
    conex = []

    done = False
    observation = env.reset()
    n_steps = 0
    score = 0
    route.append(env.get_agentPos())
    conex.append(env.is_conectado())

    while not done:
        action = choose_action(model, observation)
        observation_, reward, done = env.step(action)
        score += reward

        observation = observation_
        n_steps += 1

        route.append(env.get_agentPos())
        conex.append(env.is_conectado())

    print(f'Teste do episódio {0}: Recompensa Total = {score} - Numero de Passos = {n_steps}')
    print(f'Caminho: {route}')
    print(f'Conexões: {conex}')

    return route, conex

def plotar_caminho(env, posicoes_agentes, historico_conexao, caminho=False):
    name = 'environment.png'
    title = 'Exploração do Ambiente'
    if (caminho):
        name =  'best_route.png'
        title = 'Melhor Caminho'

    plt.figure(figsize=(12, 12))

    antenas = np.array(env.get_antennas())
    plt.scatter(antenas[:, 0], antenas[:, 1], marker="D", color='#9000ff', s=500, label='Antenas')

    plt.scatter(*env.ponto_origem, marker=6, color='#ffff00',s=100, label='Ponto de Origem')
    plt.scatter(*env.ponto_destino, marker=6, color='#a3a305',s=100, label='Ponto de Destino')

    cores = ['green' if conectado else 'red' for conectado in historico_conexao]
    plt.scatter(*zip(*posicoes_agentes), c=cores, marker='o', label='Conecxoes')

    if (caminho):
        plt.plot(*zip(*posicoes_agentes), linestyle='-', color='gray', linewidth=1)

    legendas = [
        plt.Line2D([0], [0], marker="D", color='w', markerfacecolor='#9000ff', markersize=8, label='Antenas'),
        plt.Line2D([0], [0], marker=6, color='w', markerfacecolor='#ffff00', markersize=8, label='Ponto de Origem'),
        plt.Line2D([0], [0], marker=6, color='w', markerfacecolor='#a3a305', markersize=8, label='Ponto de Destino'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Conectado'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Desconectado')
    ]

    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(handles=legendas, loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)

    plt.savefig('plots/'+ env.name + '_' + name)
