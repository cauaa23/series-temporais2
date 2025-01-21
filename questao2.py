import numpy as np

# Coeficientes do modelo AR
phi = np.array([2.1, -1.47, 0.343])

# Variância do ruído branco (assumindo σ^2_e = 1)
sigma_e2 = 1

# Sistema de equações de Yule-Walker para AR(3)
# A matriz de coeficientes (AR)
R = np.array([
    [1, phi[0], phi[1], phi[2]],
    [phi[0], 1, phi[0], phi[1]],
    [phi[1], phi[0], 1, phi[0]],
    [phi[2], phi[1], phi[0], 1]
])

# Vetor de autocovariâncias (gammas)
gamma = np.array([sigma_e2, 0, 0, 0])

# Resolver o sistema linear para as autocovariâncias
autocovariances = np.linalg.solve(R, gamma)

# A autocovariância teórica
print("Autocovariâncias teóricas:", autocovariances)
