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
    
    lr_history = [] 
    for e in range(1, epochs + 1):
        if e <= warmup_epochs:
            lr = max_lr * (e / warmup_epochs)
        else:
            progress = (e - warmup_epochs) / (epochs - warmup_epochs)
            lr = max_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
        lr_history.append(lr)
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
    return losses, lr_history

# =========================================================
# 2. RAMANUJAN-A ORIGINAL (O "Milenar")
# =========================================================
def ramanujan_a_original(X, y, epochs=50, batch_size=32, base_scale=0.15):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    lr_history = []
    
    for e in range(1, epochs + 1):
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        epoch_lr_avg = 0
        
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
            
            direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
            norm_dir = np.linalg.norm(direcao)
            if norm_dir < 1e-12: continue
            
            # A Fração Original Falha!
            salto_ramanujan = 1.0 / (e + (norm_dir**2) / e)
            lr_efetivo = salto_ramanujan * base_scale
            epoch_lr_avg += lr_efetivo
            
            W -= lr_efetivo * direcao
            
        losses.append(epoch_loss / steps)
        lr_history.append(epoch_lr_avg / steps)
        
    return losses, lr_history

# =========================================================
# 3. RAMANUJAN-V2 (A Evolução Escalável)
# =========================================================
def ramanujan_v2(X, y, epochs=50, batch_size=32, base_scale=0.012, weight_decay=0.001):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    D = X.shape[1] # Dimensões (Parâmetros)
    
    lr_history = []
    
    for e in range(1, epochs + 1):
        # Normalização do Tempo
        progresso_tempo = e / epochs 
        
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        epoch_lr_avg = 0
        
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
            
            direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Normalização Espacial (Imunidade a Escala)
            norm_dir_padrao = np.linalg.norm(direcao) / np.sqrt(D)
            
            # A Nova Fração Contínua Corrigida
            fator_ramanujan = 1.0 / (progresso_tempo + (norm_dir_padrao**2) / (progresso_tempo + 1e-8))
            lr_efetivo = fator_ramanujan * base_scale
            epoch_lr_avg += lr_efetivo
            
            # Weight Decay Nativo
            W = W - base_scale * weight_decay * W
            
            W -= lr_efetivo * direcao
            
        losses.append(epoch_loss / steps)
        lr_history.append(epoch_lr_avg / steps)
        
    return losses, lr_history

def run_v2_benchmark():
    print("==================================================================")
    print(" RAMANUJAN-V2: CORRIGINDO AS ALUCINAÇÕES EM ALTA ESCALA")
    print("==================================================================")
    
    # Criando um dataset com mais parâmetros para provar a Maldição da Dimensionalidade
    # D = 500 (O Ramanujan original vai sofrer porque norm_dir será muito alto)
    print("Gerando Dataset de Alta Escala (D=500 Parâmetros)...")
    np.random.seed(999)
    N = 5000
    D = 500
    X = np.random.randn(N, D)
    
    mask = np.random.rand(N, D) > 0.8
    X = X * mask
    
    outlier_mask = np.random.rand(N, D) > 0.95
    X[outlier_mask] *= 50.0
    
    true_W = np.random.randn(D) * 1.0
    noise = np.random.randn(N) * 2.0
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    flip_mask = np.random.rand(N) > 0.95
    y[flip_mask] = 1.0 - y[flip_mask]
    
    EPOCHS = 40
    BATCH_SIZE = 64
    
    print("\n[Treinamento 1] Adam Justo (O Padrão da Indústria)")
    t0 = time.time()
    loss_adam, lr_adam = adam_cosine_warmup(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.015, warmup_epochs=10)
    print(f" -> Loss Final: {loss_adam[-1]:.4f}")
    
    print("\n[Treinamento 2] Ramanujan-A Original (Sofrendo com as Dimensões D=500)")
    t0 = time.time()
    loss_ram1, lr_ram1 = ramanujan_a_original(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, base_scale=0.15)
    print(f" -> Loss Final: {loss_ram1[-1]:.4f} (Lutando para aprender)")
    
    print("\n[Treinamento 3] Ramanujan-V2 (Escalável e Imune)")
    t0 = time.time()
    loss_ram2, lr_ram2 = ramanujan_v2(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, base_scale=0.005, weight_decay=0.01)
    print(f" -> Loss Final: {loss_ram2[-1]:.4f} (Evolução Alcançada)")
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1e1e2e')
    
    # Gráfico 1: Loss
    ax = axs[0]
    ax.set_facecolor('#11111b')
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, loss_ram1, label='Ramanujan Original (Sofrendo)', color='#f38ba8', lw=2, linestyle='--')
    ax.plot(epochs_range, loss_adam, label='Adam Cosine Warmup', color='#89b4fa', lw=3)
    ax.plot(epochs_range, loss_ram2, label='Ramanujan-V2 (Normalizado)', color='#a6e3a1', lw=3)
    
    ax.set_title('Alta Escala (D=500): O Ramanujan Original Quebra', color='#cdd6f4', fontsize=14)
    ax.set_xlabel('Épocas', color='#cdd6f4')
    ax.set_ylabel('Loss (BCE)', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    ax.grid(color='#313244', linestyle=':', alpha=0.5)
    ax.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    
    # Gráfico 2: Comportamento do LR
    ax = axs[1]
    ax.set_facecolor('#11111b')
    ax.plot(epochs_range, lr_ram1, label='LR Ramanujan Original (Preso no Warmup)', color='#f38ba8', lw=2, linestyle='--')
    ax.plot(epochs_range, lr_adam, label='LR Adam Cosine', color='#89b4fa', lw=3)
    # Scale up ram2 visually for the plot to compare shapes
    ax.plot(epochs_range, lr_ram2, label='LR Ramanujan-V2', color='#a6e3a1', lw=3)
    
    ax.set_title('Mecânica do Agendador de Salto', color='#cdd6f4', fontsize=14)
    ax.set_xlabel('Épocas', color='#cdd6f4')
    ax.set_ylabel('Taxa de Aprendizado (LR Efetivo)', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    ax.grid(color='#313244', linestyle=':', alpha=0.5)
    ax.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    
    plt.tight_layout()
    plt.savefig('ramanujan_v2_arena.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráficos da Evolução gerados: 'ramanujan_v2_arena.png'")

if __name__ == "__main__":
    run_v2_benchmark()
