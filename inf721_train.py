import inf721_environment
import inf721_agent
import inf721_utils
import torch
import numpy as np

ambiente = 1 # Altere o ambiente

if (ambiente == 1):
    tamanho_ambiente = 12
    ponto_origem = (0, 0)
    ponto_destino = (9, 9)
    p_trans = 0.000003

elif ( ambiente == 2):
    tamanho_ambiente = 80
    ponto_origem = (0, 0)
    ponto_destino = (45, 60)
    p_trans = 0.0002

else:
    tamanho_ambiente = 200
    ponto_origem = (0, 0)
    ponto_destino = (175, 175)
    p_trans = 0.0007

env = inf721_environment.Environment(tamanho_ambiente, ponto_origem, ponto_destino, p_trans)

posicoes_agentes, historico_conexao = env.explorar_ambiente()
inf721_utils.plotar_caminho(env, posicoes_agentes, historico_conexao)

# Altere de acordo com o Ambiente treinado
# disponível em: 

# Taxa de Aprendizado
lr = 0.003

# Tamanho do batch
batch_size = 128

# Gama
gamma = 0.08

# Epsilon
epsilon = 1.0
eps_min = 0.01
eps_dec = 0.001

state_dim = env.get_state().shape[0]
n_actions = 4 # esquerda / direita / cima / baixo

# Numero de epocas
num_epochs = 300

agent = inf721_agent.Agent(gamma=gamma, epsilon=epsilon, learning_rate=lr,
              state_dim=state_dim, batch_size=batch_size,
              n_actions=n_actions, eps_min=eps_min, eps_dec=eps_dec,
              replace=10000, env_name=env.name)

best_score = -np.inf

scores, avg_scores, steps_array = [], [], []

for epoch in range(num_epochs):
    done = False
    observation = env.reset()
    n_steps = 0

    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done = env.step(action)
        score += reward

        agent.store_transition(observation, action, reward, observation_, done)
        agent.learn()

        observation = observation_
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)

    print('episode: ', epoch, 'score: ', score,
            ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            ' epsilon %.2f' % agent.epsilon, 'steps', n_steps)

    if score > best_score:
        best_score = score
    
    # Para o treinamento do ambiente 1 Descomente o código abaixo
    # e comente as linhas 98 à 101 em inf721_agent.py
    #
    # if agent.epsilon > agent.eps_min:
    #     agent.epsilon -= agent.eps_dec
    # else:
    #     agent.epsilon = agent.eps_min

inf721_utils.plot_train(env.name, avg_scores, steps_array)
torch.save(agent.Q_eval.state_dict(), agent.save_dir + 'DQN_wb' + agent.env_name + '.pth')


