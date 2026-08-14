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
# 1. ADAM JUSTO (O Alvo a ser batido)
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
# 2. THE TRUE OMEGA (Engenharia Crítica)
# =========================================================
def the_true_omega(X, y, epochs=50, batch_size=32, max_lr=0.02, weight_decay=0.001):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    D = X.shape[1] 
    
    # Variáveis de Topologia (Memória Basin Hopping)
    loss_ema = float('inf')
    patience = 0
    clock_reset_epoch = 0 # Para o Warm Restart do Ramanujan
    
    for e in range(1, epochs + 1):
        # 1. Relógio Interno Ramanujan (Reativo em vez de Estático)
        internal_e = e - clock_reset_epoch
        progresso = internal_e / epochs
        
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
            
            direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # --- O ENVELOPE RAMANUJAN ---
            # O Ramanujan não decide a LR absoluta mais, ele atua como um modulador de envelope
            norm_dir_padrao = np.linalg.norm(direcao) / np.sqrt(D)
            # Quanto mais turbulento (norm alto), menor o envelope. 
            envelope = 1.0 / (progresso + norm_dir_padrao + 1e-8)
            
            # LR Base é um Cosseno modulado pelo envelope Ramanujan
            fator = 0.5 * (1 + math.cos(math.pi * progresso))
            lr_efetivo = max_lr * min(envelope * fator, 1.5) # Limitado a 1.5x do max_lr
            
            a = lr_efetivo * direcao
            
            # --- O FREIO BAKHSHALI INTELIGENTE (Gate) ---
            # Em vez de frear a rede inteira (o que falhou antes), só usamos a matemática antiga
            # de Bakhshali nas dimensões onde o gradiente está tendo uma explosão estatística!
            # Identificamos outliers: gradiente atual é 3x maior que a média histórica (sqrt(v_hat))
            explosao_mask = (np.abs(grad) > 3.0 * np.sqrt(v_hat))
            
            denom = np.abs(W) + np.abs(a) + 1e-8
            correction = (a ** 2) / (2.0 * denom)
            
            # Aplica Bakhshali SÓ nos parâmetros em pânico. O resto segue o Adam normalmente.
            step_size = np.where(explosao_mask, a - np.sign(a) * correction, a)
            
            W = W - lr_efetivo * weight_decay * W
            W -= step_size
            
        avg_loss = epoch_loss / steps
        losses.append(avg_loss)
        
        # --- OMNI BASIN HOPPING (Detecção de Estagnação) ---
        if e == 1:
            loss_ema = avg_loss
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * avg_loss
            
        # Se a rede ficou presa num platô no meio do caos...
        if avg_loss > loss_ema * 0.99:
            patience += 1
        else:
            patience = 0
            
        if patience >= 4 and clock_reset_epoch < e:
            # Teleport Kick (Ruído Térmico Localizado para pular fora do Mínimo Local)
            W += np.random.randn(D) * 0.05 * np.std(W)
            patience = 0
            clock_reset_epoch = e # Isso dá um "Warm Restart" no Envelope Ramanujan na próxima época!
            
    return losses

def run_true_omega_arena():
    print("==================================================================")
    print(" A NOVA ENGENHARIA: O VERDADEIRO ÔMEGA VS ADAM SOTA")
    print("==================================================================")
    
    np.random.seed(888)
    N = 8000
    D = 1000
    X = np.random.randn(N, D)
    
    mask = np.random.rand(N, D) > 0.9
    X = X * mask
    
    outlier_mask = np.random.rand(N, D) > 0.90
    X[outlier_mask] *= 500.0
    
    true_W = np.random.randn(D) * 1.5
    noise = np.random.randn(N) * 5.0
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    flip_mask = np.random.rand(N) > 0.90
    y[flip_mask] = 1.0 - y[flip_mask]
    
    EPOCHS = 50
    BATCH_SIZE = 128
    
    print("\n[Lutador 1] Adam Cosine Warmup")
    t0 = time.time()
    loss_adam = adam_cosine_warmup(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.03, warmup_epochs=10)
    print(f" -> Loss Final: {loss_adam[-1]:.4f} | Tempo: {time.time()-t0:.2f}s")
    
    print("\n[Lutador 2] The True Omega (Ramanujan Dinâmico + Bakhshali Gate + OMNI Kick)")
    t0 = time.time()
    loss_omega = the_true_omega(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.03, weight_decay=0.01)
    print(f" -> Loss Final: {loss_omega[-1]:.4f} | Tempo: {time.time()-t0:.2f}s")
    
    # Plotting
    plt.figure(figsize=(10, 7), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, loss_adam, label='Adam SOTA (O Rei)', color='#f38ba8', lw=2, linestyle='--')
    plt.plot(epochs_range, loss_omega, label='The True Omega (A Criatividade)', color='#a6e3a1', lw=3)
    
    plt.title('A Revanche da Inteligência: Engenharia Crítica em Ação', color='#cdd6f4', fontsize=16)
    plt.xlabel('Épocas de Treinamento', color='#cdd6f4', fontsize=14)
    plt.ylabel('Erro Absoluto (BCE Loss)', color='#cdd6f4', fontsize=14)
    
    plt.ylim(min(loss_omega[-1], loss_adam[-1]) - 0.05, max(loss_adam)*1.1)
    
    plt.tick_params(axis='x', colors='#cdd6f4', labelsize=12)
    plt.tick_params(axis='y', colors='#cdd6f4', labelsize=12)
    plt.grid(True, color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('the_true_omega_arena.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráfico da Vitória gerado: 'the_true_omega_arena.png'")

if __name__ == "__main__":
    run_true_omega_arena()
