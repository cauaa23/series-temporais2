import numpy as np
import statsmodels.api as sm

# Gerar os dados do processo AR conforme o código anterior
n_samples = 1000
phi = [2.1, -1.47, 0.343]  # Coeficientes do modelo AR(3)
e_t = np.random.normal(0, 1, n_samples)  # Ruído branco
nu_t = np.zeros(n_samples)

# Simular o processo AR(3)
for t in range(3, n_samples):
    nu_t[t] = phi[0] * nu_t[t-1] + phi[1] * nu_t[t-2] + phi[2] * nu_t[t-3] + e_t[t]

# Preparar os dados para regressão: X será uma matriz com as defasagens
X = np.column_stack([nu_t[2:-1], nu_t[1:-2], nu_t[:-3]])  # Defasagens de 1, 2 e 3
y = nu_t[3:]  # A variável dependente (ν_t)

# Adicionar o termo de intercepto
X = sm.add_constant(X)

# Ajustar o modelo de regressão linear
model = sm.OLS(y, X).fit()

# Exibir os parâmetros estimados
print("Parâmetros estimados do modelo AR(3):")
print(model.summary())


#questao4.2

import numpy as np
import matplotlib.pyplot as plt

# Gerar os dados do processo AR conforme o código anterior
n_samples = 1000
phi_true = [2.1, -1.47, 0.343]  # Coeficientes do modelo AR(3)
e_t = np.random.normal(0, 1, n_samples)  # Ruído branco
nu_t = np.zeros(n_samples)

# Simular o processo AR(3)
for t in range(3, n_samples):
    nu_t[t] = phi_true[0] * nu_t[t - 1] + phi_true[1] * nu_t[t - 2] + phi_true[2] * nu_t[t - 3] + e_t[t]


# Função para calcular as autocovariâncias
def autocovariance(x, lag):
    N = len(x)
    mu = np.mean(x)

    if lag == 0:
        return np.var(x)

    # Garantir que o índice de corte não ultrapasse o tamanho do array
    return np.sum((x[lag:] - mu) * (x[:-lag] - mu)) / N


# Calcular as autocovariâncias
max_lag = 3  # Vamos estimar o AR(3), então usamos até o lag 3
autocovariances = [autocovariance(nu_t, h) for h in range(max_lag + 1)]

# Construir o sistema de Yule-Walker
# O vetor de autocovariâncias (γ) é [γ_0, γ_1, γ_2]
gamma = autocovariances

# A matriz de autocovariâncias (R) será 3x3 (AR(3))
R = np.zeros((max_lag, max_lag))
for i in range(max_lag):
    for j in range(max_lag):
        R[i, j] = autocovariance(nu_t, abs(i - j))

# Resolver as equações de Yule-Walker
phi_yw = np.linalg.solve(R, gamma[1:])

# Exibir os parâmetros estimados usando Yule-Walker
print("Parâmetros estimados do modelo AR(3) usando Yule-Walker:")
print(phi_yw)

# Verificar os parâmetros verdadeiros
print("\nParâmetros verdadeiros do modelo AR(3):")
print(phi_true)

# Plotar a autocovariância (sem o argumento 'use_line_collection')
plt.figure(figsize=(10, 6))
plt.stem(range(max_lag + 1), autocovariances, basefmt=" ")
plt.title("Autocovariâncias")
plt.xlabel("Lag")
plt.ylabel("Autocovariância")
plt.show()
