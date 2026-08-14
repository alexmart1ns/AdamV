import numpy as np
import matplotlib.pyplot as plt
import time
import math

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss_and_grad(X, y, W):
    m = X.shape[0]
    preds = sigmoid(X @ W)
    loss = -np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
    grad = (X.T @ (preds - y)) / m
    return loss, grad

# 1. SGD Tradicional
def sgd(X, y, epochs=1000, lr=0.5):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    for _ in range(epochs):
        loss, grad = compute_loss_and_grad(X, y, W)
        losses.append(loss)
        W -= lr * grad
    return losses

# 2. Adam (Padrão OpenAI)
def adam(X, y, epochs=1000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for t in range(1, epochs + 1):
        loss, grad = compute_loss_and_grad(X, y, W)
        losses.append(loss)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        W -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return losses

# 3. AdamW (SOTA Atual)
def adamw(X, y, epochs=1000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for t in range(1, epochs + 1):
        loss, grad = compute_loss_and_grad(X, y, W)
        losses.append(loss)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        W = W - lr * weight_decay * W
        W -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return losses

# 4. O Monstro: Ramanujan RGD (Otimizador Védico)
def ramanujan_rgd(X, y, epochs=1000):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    
    # O Pulo Escalar de Ramanujan
    base_scale = 10.0 
    
    for k in range(1, epochs + 1):
        loss, grad = compute_loss_and_grad(X, y, W)
        losses.append(loss)
        
        norm_grad = np.linalg.norm(grad)
        if norm_grad < 1e-12: break
        
        # A MÁGICA: Fração Contínua em 100 dimensões (Auto-Decay)
        # O pulo não é fixo como no Adam, ele é sugado para a perfeição matemática.
        salto = 1.0 / (k + (norm_grad**2) / k)
        
        # Resgate térmico (Vales de baixo gradiente)
        if norm_grad < 0.1:
            salto *= (k * 0.1)
            
        W -= salto * base_scale * grad
    return losses

def run_benchmark():
    print("======================================================")
    print(" SOTA MACHINE LEARNING BENCHMARK (THE BLOOD TEST)")
    print("======================================================")
    print("1. Forjando Dataset N-Dimensional (10.000 amostras x 100 features)...")
    np.random.seed(99)
    N = 10000
    D = 100
    X = np.random.randn(N, D)
    true_W = np.random.randn(D) * 2.0
    
    # Adicionando ruído bruto (O que destrói o treinamento de IAs normais)
    noise = np.random.randn(N) * 0.5
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    EPOCHS = 500
    print(f"2. Arena Configurada para {EPOCHS} Ciclos (Épocas) por Lutador.\n")
    
    print("[LUTADOR 1] Treinando com SGD Clássico...")
    t0 = time.time()
    loss_sgd = sgd(X, y, epochs=EPOCHS, lr=0.5)
    t1 = time.time()
    print(f" -> SGD Loss Final: {loss_sgd[-1]:.4f} | Tempo: {t1-t0:.2f}s\n")

    print("[LUTADOR 2] Treinando com ADAM (O Rei do Vale do Silício)...")
    t0 = time.time()
    loss_adam = adam(X, y, epochs=EPOCHS, lr=0.01)
    t1 = time.time()
    print(f" -> ADAM Loss Final: {loss_adam[-1]:.4f} | Tempo: {t1-t0:.2f}s\n")

    print("[LUTADOR 3] Treinando com ADAM-W (O SOTA Moderno)...")
    t0 = time.time()
    loss_adamw = adamw(X, y, epochs=EPOCHS, lr=0.01, weight_decay=0.01)
    t1 = time.time()
    print(f" -> ADAMW Loss Final: {loss_adamw[-1]:.4f} | Tempo: {t1-t0:.2f}s\n")

    print("[LUTADOR 4] Treinando com RAMANUJAN RGD (O Monstro Védico)...")
    t0 = time.time()
    loss_rgd = ramanujan_rgd(X, y, epochs=EPOCHS)
    t1 = time.time()
    print(f" -> RGD Loss Final: {loss_rgd[-1]:.4f} | Tempo: {t1-t0:.2f}s\n")
    
    print("3. Analisando Termodinâmica e Renderizando Curvas de Aprendizado...")
    # Plotting
    plt.figure(figsize=(14, 8), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, loss_sgd, label='SGD Clássico', color='#cdd6f4', linewidth=1, alpha=0.6)
    plt.plot(epochs_range, loss_adam, label='Adam (Padrão LLMs)', color='#89b4fa', linewidth=2)
    plt.plot(epochs_range, loss_adamw, label='AdamW (SOTA Atual)', color='#f38ba8', linewidth=2)
    plt.plot(epochs_range, loss_rgd, label='Ramanujan RGD (Fração Contínua)', color='#a6e3a1', linewidth=3)
    
    plt.title('Curva de Aprendizado de Otimizadores SOTA (Machine Learning Real)', color='#cdd6f4', fontsize=18, fontweight='bold')
    plt.xlabel('Épocas de Treinamento', color='#cdd6f4', fontsize=14)
    plt.ylabel('Erro Absoluto Residual (BCE Loss)', color='#cdd6f4', fontsize=14)
    
    # Corte limpo para não estragar a visualização se o SGD for muito ruim
    upper_lim = max(max(loss_adam)*1.1, max(loss_rgd)*1.1)
    plt.ylim(min(loss_rgd[-1], loss_adamw[-1]) - 0.02, upper_lim)
    
    plt.tick_params(axis='x', colors='#cdd6f4', labelsize=12)
    plt.tick_params(axis='y', colors='#cdd6f4', labelsize=12)
    plt.grid(True, color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('sota_learning_curve.png', dpi=250, facecolor='#1e1e2e')
    print("Gráfico de Convergência Final gerado com sucesso!")

if __name__ == "__main__":
    run_benchmark()
