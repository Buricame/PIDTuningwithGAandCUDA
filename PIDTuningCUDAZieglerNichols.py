import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from numba import cuda, float32

# Veriyi yükleme fonksiyonu
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df['Tar'].values, df['Act'].values

# CUDA kernel fonksiyonu: PID simülasyonu
@cuda.jit
def pid_kernel(p, i, d, setpoint, actual, output, integral, last_error):
    idx = cuda.grid(1)
    if idx < actual.shape[0]:
        error = setpoint[idx] - actual[idx]
        integral[idx] += error
        derivative = error - last_error[idx]
        output[idx] = p * error + i * integral[idx] + d * derivative
        last_error[idx] = error

# PID simülasyonu ve hata hesaplama
def run_pid(params, setpoint, actual):
    p, i, d = params
    
    if not (0.01 <= p <= 5.0 and 0.01 <= i <= 2.0 and 0.001 <= d <= 0.05):
        return 1e18  # Çok yüksek bir hata değeri
    
    # CUDA için veri hazırlığı
    threads_per_block = 256
    blocks_per_grid = (len(actual) + threads_per_block - 1) // threads_per_block
    
    setpoint_device = cuda.to_device(setpoint)
    actual_device = cuda.to_device(actual)
    output_device = cuda.device_array_like(actual)
    integral_device = cuda.device_array(len(actual), dtype=np.float32)
    last_error_device = cuda.device_array(len(actual), dtype=np.float32)
    
    integral_device[:] = 0.0
    last_error_device[:] = 0.0
    
    pid_kernel[blocks_per_grid, threads_per_block](p, i, d, setpoint_device, actual_device, output_device, integral_device, last_error_device)
    
    output_host = output_device.copy_to_host()
    iae = np.sum(np.abs(setpoint - output_host))
    
    return iae

# Ziegler-Nichols yöntemi ile PID parametrelerini hesaplama
def ziegler_nichols(K_u, P_u):
    K_p = 0.6 * K_u
    T_i = P_u / 2
    T_d = P_u / 8
    K_i = K_p / T_i
    K_d = K_p * T_d
    return K_p, K_i, K_d

# Fitness fonksiyonu: PID parametrelerinin değerlendirilmesi
def evaluate(individual):
    p, i, d = individual
    iae = run_pid((p, i, d), setpoint, actual)
    print(f"IAE: {iae}, P: {p}, I: {i}, D: {d}")
    return iae,

# Mutasyon fonksiyonu: PID parametrelerinde rastgele değişiklikler yapma
def mutate(individual):
    if random.random() < MUTPB:
        idx = random.randint(0, 2)
        if idx == 0:
            individual[idx] += random.uniform(-P_INC, P_INC)
            individual[idx] = min(max(individual[idx], P_MIN), P_MAX)
        elif idx == 1:
            individual[idx] += random.uniform(-I_INC, I_INC)
            individual[idx] = min(max(individual[idx], I_MIN), I_MAX)
        elif idx == 2:
            individual[idx] += random.uniform(-D_INC, D_INC)
            individual[idx] = min(max(individual[idx], D_MIN), D_MAX)
    return individual,

# Parametreler
POP_SIZE = 50
NGEN = 25
CXPB = 0.5
MUTPB = 0.2

# PID parametreleri için aralıklar
P_MIN, P_MAX, P_INC = 0.01, 5.0, 0.005
I_MIN, I_MAX, I_INC = 0.01, 2.0, 0.01
D_MIN, D_MAX, D_INC = 0.001, 0.05, 0.001

# Genetik algoritma ayarları
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_p", random.uniform, P_MIN, P_MAX)
toolbox.register("attr_i", random.uniform, I_MIN, I_MAX)
toolbox.register("attr_d", random.uniform, D_MIN, D_MAX)
toolbox.register("individual", tools.initCycle, creator.Individual, (toolbox.attr_p, toolbox.attr_i, toolbox.attr_d), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", mutate)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    global setpoint, actual
    setpoint, actual = load_data('RatePitchNew.csv')
    
    # Ziegler-Nichols ile başlangıç PID değerlerini hesapla
    K_u = 0.5
    P_u = 0.3
    zn_p, zn_i, zn_d = ziegler_nichols(K_u, P_u)
    
    # Başlangıç popülasyonu oluştur
    population = []
    for _ in range(POP_SIZE):
        if random.random() < 0.5:
            p = zn_p + random.uniform(-P_INC, P_INC)
            i = zn_i + random.uniform(-I_INC, I_INC)
            d = zn_d + random.uniform(-D_INC, D_INC)
        else:
            p = random.uniform(P_MIN, P_MAX)
            i = random.uniform(I_MIN, I_MAX)
            d = random.uniform(D_MIN, D_MAX)
        individual = creator.Individual([p, i, d])
        population.append(individual)
    
    # Genetik algoritmayı çalıştır
    algorithms.eaSimple(population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, verbose=True)

    # En iyi sonucu yazdır
    best_individual = tools.selBest(population, 1)[0]
    print("En iyi PID parametreleri:")
    print(f"P: {best_individual[0]:.3f}")
    print(f"I: {best_individual[1]:.3f}")
    print(f"D: {best_individual[2]:.3f}")

if __name__ == "__main__":
    main()
