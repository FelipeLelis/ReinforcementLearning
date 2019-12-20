import gym

#ambiente = gym.make('MountainCarContinuous-v0')
#ambiente = gym.make('CartPole-v0')
ambiente = gym.make('Acrobot-v1')
ambiente.reset()

for _ in range(500):
    ambiente.render()
    acao = ambiente.action_space.sample()
    observacao, recompensa, finalizado, informacoes = ambiente.step(acao)
    print(observacao, recompensa, finalizado, informacoes)
