import numpy as np
import matplotlib.pyplot as plt

#questao1a
# Gerar ruído branco com média 0 e desvio padrão 1
et = np.random.normal(0, 1, 1000)

# Exibir as 10 primeiras amostras como exemplo
print(et[:10])


#questao1b

# Número de amostras
n_samples = 1000

# Coeficientes do modelo AR
a = [2.1, -1.47, 0.343]

# Gerar ruído branco (e_t) com média 0 e desvio padrão 1
e_t = np.random.normal(0, 1, n_samples)

# Inicializar o vetor para o processo AR
nu_t = np.zeros(n_samples)

# Simular o processo AR
for t in range(3, n_samples):
    nu_t[t] = a[0] * nu_t[t-1] + a[1] * nu_t[t-2] + a[2] * nu_t[t-3] + e_t[t]

# Plotar o processo gerado
plt.plot(nu_t)
plt.title("Simulação do Processo AR")
plt.xlabel("Tempo (t)")
plt.ylabel("Valor de ν_t")
plt.show()
