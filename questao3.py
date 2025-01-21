import numpy as np
import matplotlib.pyplot as plt

# Gerar os dados do processo AR conforme o código anterior
n_samples = 1000
phi = [2.1, -1.47, 0.343]
e_t = np.random.normal(0, 1, n_samples)
nu_t = np.zeros(n_samples)

# Simular o processo AR
for t in range(3, n_samples):
    nu_t[t] = phi[0] * nu_t[t-1] + phi[1] * nu_t[t-2] + phi[2] * nu_t[t-3] + e_t[t]

# Função para calcular a autocovariância para um lag h
def autocovariance(x, h):
    N = len(x)
    mu = np.mean(x)
    if h == 0:  # Caso especial para lag 0
        return np.var(x)
    else:
        return np.sum((x[h:] - mu) * (x[:-h] - mu)) / (N - h)

# Calcular as autocovariâncias para diferentes lags
max_lag = 30  # Número máximo de lags para calcular
autocovariances = [autocovariance(nu_t, h) for h in range(max_lag + 1)]

# Calcular a autocorrelação
gamma_0 = autocovariances[0]  # Autocovariância no lag 0
autocorrelation = [gamma / gamma_0 for gamma in autocovariances]

# Exibir resultados
print("Autocovariâncias:", autocovariances)
print("Autocorrelações:", autocorrelation)

# Plotar a autocovariância e a autocorrelação
plt.figure(figsize=(12, 6))

# Plotar a autocovariância
plt.subplot(1, 2, 1)
plt.stem(range(max_lag + 1), autocovariances, basefmt=" ")  # Removido 'use_line_collection=True'
plt.title("Autocovariâncias")
plt.xlabel("Lag")
plt.ylabel("Autocovariância")

# Plotar a autocorrelação
plt.subplot(1, 2, 2)
plt.stem(range(max_lag + 1), autocorrelation, basefmt=" ")  # Removido 'use_line_collection=True'
plt.title("Autocorrelações")
plt.xlabel("Lag")
plt.ylabel("Autocorrelação")

plt.tight_layout()
plt.show()
