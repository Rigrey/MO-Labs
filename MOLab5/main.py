import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style

Xmin, Xmax = 0, np.pi
K = 100
a = 0.25
L = 10
r_values = [3, 5]

x_k = np.linspace(Xmin, Xmax, K + 1)
f_k = np.sin(x_k) + 0.5

np.random.seed(42)
noise = np.random.uniform(-a, a, len(f_k))
f_noisy = f_k + noise

def geometric_mean_filter(signal, r, weights):
    filtered_signal = []
    n = len(signal)
    for k in range(n):
        num, den = 1, 0  # Инициализация для геометрического среднего
        for i in range(-r+1, r):
            idx = k + i
            if 0 <= idx < n:
                w = weights[abs(i)]
                if signal[idx] > 0:  # Избегаем логарифмов от отрицательных значений или нуля
                    num *= signal[idx] ** w
                    den += w
        filtered_signal.append(num ** (1 / den) if den != 0 else 0)  # Геометрическое среднее
    return np.array(filtered_signal)

def compute_chebyshev_metrics(original, noisy, filtered):
    omega = np.max(np.abs(filtered[:-1] - filtered[1:]))  # max(|f_k - f_{k-1}|)
    delta = np.max([np.abs(filtered[k] - noisy[k]) for k in range(len(original))])  # max(|f_noisy_k - f_original_k|)
    dist = max(omega, delta)
    return omega, delta, dist


def random_search(signal, noisy_signal, r, n_iter=100):
    best_weights = None
    best_dist = float("inf")
    for _ in range(n_iter):
        weights = np.random.rand(r)
        weights = weights / np.sum(weights)  # Нормализация
        filtered_signal = geometric_mean_filter(noisy_signal, r, weights)
        omega, delta, dist = compute_chebyshev_metrics(signal, noisy_signal, filtered_signal)
        if dist < best_dist:
            best_dist = dist
            best_weights = weights
    return best_weights, best_dist

def passive_search_criteria(signal, noisy_signal, r, L=10):
    lambdas = [l / L for l in range(L + 1)]
    omega_values = []
    delta_values = []
    dist_values = []
    weights_values = []

    for lamb in lambdas:
        weights = np.random.rand(r)
        weights = weights / np.sum(weights)  # Нормализация
        filtered_signal = geometric_mean_filter(noisy_signal, r, weights)
        omega, delta, dist = compute_chebyshev_metrics(signal, noisy_signal, filtered_signal)
        omega_values.append(round(omega, 3))
        delta_values.append(round(delta, 3))
        dist_values.append(round(dist, 3))
        weights_values.append(np.round(weights, 3))

    return lambdas, omega_values, delta_values, dist_values, weights_values


for r in r_values:
    best_weights, best_dist_random = random_search(f_k, f_noisy, r)
    filtered_signal_random = geometric_mean_filter(f_noisy, r, best_weights)

    lambdas, omega_values, delta_values, dist_values, weights_values = passive_search_criteria(f_k, f_noisy, r)

    best_dist = min(dist_values)
    best_lambda_index = dist_values.index(best_dist)
    best_lambda = lambdas[best_lambda_index]
    weights_passive = weights_values[best_lambda_index]
    filtered_signal_passive = geometric_mean_filter(f_noisy, r, weights_passive)

    plt.figure(figsize=(12, 6))
    plt.plot(x_k, f_k, label="Исходный сигнал $f_k$", linewidth=2)
    plt.plot(x_k, f_noisy, label="Зашумленный сигнал $f'_k$", linewidth=2)
    plt.plot(x_k, filtered_signal_random, label=f"Фильтр. сигнал (Random)", linewidth=2)
    plt.plot(x_k, filtered_signal_passive, label=f"Фильтр. сигнал (Passive)", linewidth=2)
    plt.title(f"Фильтрация сигнала с окном r={r}")
    plt.xlabel("$x_k$")
    plt.ylabel("$f_k$, $f'_k$")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(omega_values, delta_values, c=lambdas, cmap='viridis', s=50, edgecolor='k', label="$\\lambda$")
    for i, lamb in enumerate(lambdas):
        plt.annotate(f"{lamb:.2f}", (omega_values[i], delta_values[i]), textcoords="offset points", xytext=(5, 5),
                     ha='center')

    plt.scatter(omega_values[best_lambda_index], delta_values[best_lambda_index],
                c='gold', s=200, edgecolor='black', label=f"Лучшее $\\lambda$ = {best_lambda:.2f}")
    plt.scatter(0, 0, c='red', s=100, edgecolor='k', label="Utopia $(0, 0)$")
    plt.title(f"Критерии $\\omega$ и $\\delta$ для $r={r}$")
    plt.xlabel("Критерий $\\omega$ (макс. изменение фильтрованного сигнала)")
    plt.ylabel("Критерий $\\delta$ (макс. отклонение)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Окно r={r}:")
    max_weights_width = max(len(str(weight)) for weight in weights_values)
    header = f" λ   | dist  | {'weights'.ljust(max_weights_width)} | ω     | δ    "
    print(header)
    print("-" * len(header))
    for i in range(11):
        if i == best_lambda_index:
            print(
                Fore.GREEN +
                f"{lambdas[i]:.2f} | {dist_values[i]:.3f} | {str(weights_values[i]).ljust(max_weights_width)} | {omega_values[i]:.3f} | {delta_values[i]:.3f}"
                + Style.RESET_ALL
            )
        else:
            print(
                f"{lambdas[i]:.2f} | {dist_values[i]:.3f} | {str(weights_values[i]).ljust(max_weights_width)} | {omega_values[i]:.3f} | {delta_values[i]:.3f}"
            )
    print(
        f"Лучший λ* = {best_lambda:.2f}, dist = {best_dist:.3f}, ω = {omega_values[best_lambda_index]:.3f}, δ = {delta_values[best_lambda_index]:.3f}"
    )
    print()
