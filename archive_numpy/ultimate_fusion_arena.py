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

# =========================================================
# 1. ADAM JUSTO (Cosine + Warmup Real)
# =========================================================
def adam_cosine_warmup(X, y, epochs=50, batch_size=32, max_lr=0.01, warmup_epochs=10):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    
    for e in range(1, epochs + 1):
        if e <= warmup_epochs:
            lr = max_lr * (e / warmup_epochs)
        else:
            progress = (e - warmup_epochs) / (epochs - warmup_epochs)
            lr = max_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        for i in range(0, N, batch_size):
            t += 1
            idx = indices[i:i+batch_size]
            X_b, y_b = X[idx], y[idx]
            
            loss, grad = compute_loss_and_grad(X_b, y_b, W)
            epoch_loss += loss
            steps += 1
            
            m_v = 0.9 * m_v + 0.1 * grad
            v_v = 0.999 * v_v + 0.001 * (grad ** 2)
            m_hat = m_v / (1 - 0.9 ** t)
            v_hat = v_v / (1 - 0.999 ** t)
            
            W -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)
        losses.append(epoch_loss / steps)
    return losses

# =========================================================
# 2. O ÔMEGA: RAMANUJAN-BAKHSHALI (RB-Optimizer)
# =========================================================
def ramanujan_bakhshali(X, y, epochs=50, batch_size=32, base_scale=0.012, weight_decay=0.001):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    D = X.shape[1] 
    
    for e in range(1, epochs + 1):
        progresso_tempo = e / epochs 
        
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        
        for i in range(0, N, batch_size):
            t += 1
            idx = indices[i:i+batch_size]
            X_b, y_b = X[idx], y[idx]
            
            loss, grad = compute_loss_and_grad(X_b, y_b, W)
            epoch_loss += loss
            steps += 1
            
            # Motor Principal: Bússola do Adam
            m_v = 0.9 * m_v + 0.1 * grad
            v_v = 0.999 * v_v + 0.001 * (grad ** 2)
            m_hat = m_v / (1 - 0.9 ** t)
            v_hat = v_v / (1 - 0.999 ** t)
            
            direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # 1. ESCALONADOR: Ramanujan-V2 (Imunidade Dimensional e Temporal)
            norm_dir_padrao = np.linalg.norm(direcao) / np.sqrt(D)
            fator_ramanujan = 1.0 / (progresso_tempo + (norm_dir_padrao**2) / (progresso_tempo + 1e-8))
            lr_efetivo = fator_ramanujan * base_scale
            
            # Passo Teórico
            a = lr_efetivo * direcao
            
            # 2. FREIO ABS: Amortecedor Quártico de Bakhshali
            denom = np.abs(W) + np.abs(a) + 1e-8
            correction = (a ** 2) / (2.0 * denom)
            
            # O Passo Final Ajustado (Com Amortecimento de Overshooting)
            step_size = a - np.sign(a) * correction
            
            # 3. Integração do AdamW (Weight Decay)
            W = W - base_scale * weight_decay * W
            
            # Aplica o passo real
            W -= step_size
            
        losses.append(epoch_loss / steps)
        
    return losses

def run_ultimate_arena():
    print("==================================================================")
    print(" A BATALHA FINAL: ADAM SOTA VS RAMANUJAN-BAKHSHALI (ÔMEGA)")
    print("==================================================================")
    
    # Gerando o Dataset do Pior Cenário (Extremo Ruído, Alta Dimensionalidade e Esparsidade)
    print("Invocando a Tempestade de Ruído (D=1000 Parâmetros, Outliers Extremos)...")
    np.random.seed(777)
    N = 8000
    D = 1000
    X = np.random.randn(N, D)
    
    # 90% Esparsidade
    mask = np.random.rand(N, D) > 0.9
    X = X * mask
    
    # Outliers absurdos em 10% dos dados para forçar Explosão de Gradiente (Overshooting)
    outlier_mask = np.random.rand(N, D) > 0.90
    X[outlier_mask] *= 500.0
    
    true_W = np.random.randn(D) * 1.5
    noise = np.random.randn(N) * 5.0 # Ruído massivo nas respostas
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    # Ruído de Rótulo Bruto (10% de Mentirosos)
    flip_mask = np.random.rand(N) > 0.90
    y[flip_mask] = 1.0 - y[flip_mask]
    
    EPOCHS = 50
    BATCH_SIZE = 128
    
    print("\n[Lutador 1] Adam Cosine Warmup (O Campeão SOTA Moderno)")
    t0 = time.time()
    loss_adam = adam_cosine_warmup(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.02, warmup_epochs=10)
    print(f" -> Loss Final: {loss_adam[-1]:.4f} | Tempo: {time.time()-t0:.2f}s")
    
    print("\n[Lutador 2] Otimizador Ômega (Ramanujan-V2 + Bakhshali ABS)")
    t0 = time.time()
    loss_omega = ramanujan_bakhshali(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, base_scale=0.008, weight_decay=0.01)
    print(f" -> Loss Final: {loss_omega[-1]:.4f} | Tempo: {time.time()-t0:.2f}s")
    
    # Plotting
    plt.figure(figsize=(10, 7), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, loss_adam, label='Adam SOTA (Cosine Warmup)', color='#f38ba8', lw=2)
    plt.plot(epochs_range, loss_omega, label='Ramanujan-Bakhshali (O Ômega)', color='#a6e3a1', lw=3)
    
    plt.title('A Batalha Final no Pior Cenário Possível (Ruído e Explosão)', color='#cdd6f4', fontsize=16)
    plt.xlabel('Épocas de Treinamento', color='#cdd6f4', fontsize=14)
    plt.ylabel('Erro Absoluto (BCE Loss)', color='#cdd6f4', fontsize=14)
    
    # Escala para evidenciar o detalhe
    plt.ylim(min(loss_omega[-1], loss_adam[-1]) - 0.05, max(loss_adam)*1.1)
    
    plt.tick_params(axis='x', colors='#cdd6f4', labelsize=12)
    plt.tick_params(axis='y', colors='#cdd6f4', labelsize=12)
    plt.grid(True, color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('ultimate_fusion_arena.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráfico do Apocalipse gerado: 'ultimate_fusion_arena.png'")

if __name__ == "__main__":
    run_ultimate_arena()
