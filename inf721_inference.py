import inf721_model
import inf721_environment
import inf721_utils
import torch

ambiente = 2 # Troque aqui qual dos 3 ambientes testar


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

state_dim = env.get_state().shape[0]
n_actions = 4 

model = inf721_model.DeepQNetwork(0, state_dim, n_actions)


if (ambiente == 1):
    model.load_state_dict(torch.load('models/DQN_wb-Env_12_0-0_9-9_p3e-06.pth'))

elif ( ambiente == 2):
    model.load_state_dict(torch.load('models/DQN_wb-Env_80_0-0_45-60_p0.0002.pth'))

else:
    model.load_state_dict(torch.load('models/DQN_wb-Env_12_0-0_9-9_p3e-06.pth'))

route, conex = inf721_utils.inference_route(model, env)

inf721_utils.plotar_caminho(env, route, conex, caminho=True)
