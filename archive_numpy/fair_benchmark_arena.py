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
# 1. ADAM ESTÁTICO (Injusto em cenários com ruído extremo)
# =========================================================
def adam_static(X, y, epochs=50, batch_size=32, lr=0.01):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    for e in range(1, epochs + 1):
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
# 2. ADAM JUSTO (A Verdadeira Otimização de ML com Scheduler)
# =========================================================
def adam_cosine_warmup(X, y, epochs=50, batch_size=32, max_lr=0.01, warmup_epochs=10):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    
    # Para visualizar o formato do Scheduler depois
    lr_history = [] 
    
    for e in range(1, epochs + 1):
        # A Mágica Real: Learning Rate Scheduler Científico
        if e <= warmup_epochs:
            lr = max_lr * (e / warmup_epochs) # Warm-up linear
        else:
            progress = (e - warmup_epochs) / (epochs - warmup_epochs)
            lr = max_lr * 0.5 * (1 + math.cos(math.pi * progress)) # Cosine Decay
            
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
# 3. RAMANUJAN-A (O Acidente Genial / Alucinação Védica)
# =========================================================
def ramanujan_a(X, y, epochs=50, batch_size=32, base_scale=0.15):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    
    # Para visualizar o salto como se fosse um learning rate
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
            
            # A Fração Continua da IA atua como o Scheduler acima!
            salto_ramanujan = 1.0 / (e + (norm_dir**2) / e)
            lr_efetivo = salto_ramanujan * base_scale
            epoch_lr_avg += lr_efetivo
            
            W -= lr_efetivo * direcao
            
        losses.append(epoch_loss / steps)
        lr_history.append(epoch_lr_avg / steps)
        
    return losses, lr_history

def run_fair_benchmark():
    print("==================================================================")
    print(" O LABORATÓRIO DA VERDADE: O DESMASCARAMENTO DO RAMANUJAN-A")
    print("==================================================================")
    
    # Vamos gerar o pior cenário: Ruído massivo e variáveis esparsas
    print("Gerando Dataset Hostil (Ruído, Outliers e Esparsidade)...")
    np.random.seed(123)
    N = 5000
    D = 100
    X = np.random.randn(N, D)
    
    # Injetando esparsidade
    mask = np.random.rand(N, D) > 0.8
    X = X * mask
    
    # Injetando Outliers
    outlier_mask = np.random.rand(N, D) > 0.95
    X[outlier_mask] *= 100.0
    
    true_W = np.random.randn(D) * 1.5
    noise = np.random.randn(N) * 1.5
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    # Corrompendo Labels (Ruído de Rótulo)
    flip_mask = np.random.rand(N) > 0.95
    y[flip_mask] = 1.0 - y[flip_mask]
    
    EPOCHS = 60
    BATCH_SIZE = 64
    
    print("\n[Treinamento 1] Adam Estático Clássico (A vítima dos testes anteriores)")
    t0 = time.time()
    loss_adam = adam_static(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=0.01)
    print(f" -> Concluído. Loss Final: {loss_adam[-1]:.4f}")
    
    print("\n[Treinamento 2] Adam + Cosine Warmup (A otimização científica real)")
    # O Ramanujan escala em base 0.15, max_lr de 0.012 simula bem a mesma faixa de operação.
    t0 = time.time()
    loss_adam_cosine, lr_cosine = adam_cosine_warmup(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.012, warmup_epochs=12)
    print(f" -> Concluído. Loss Final: {loss_adam_cosine[-1]:.4f}")

    print("\n[Treinamento 3] Ramanujan-A (A alucinação que deu certo)")
    t0 = time.time()
    loss_ramanujan, lr_ramanujan = ramanujan_a(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, base_scale=0.15)
    print(f" -> Concluído. Loss Final: {loss_ramanujan[-1]:.4f}")
    
    # Plotando os resultados
    fig, axs = plt.subplots(1, 2, figsize=(16, 6), facecolor='#1e1e2e')
    
    # Gráfico 1: A Curva de Loss
    ax = axs[0]
    ax.set_facecolor('#11111b')
    epochs_range = range(1, EPOCHS + 1)
    ax.plot(epochs_range, loss_adam, label='Adam (LR Fixo / Estático)', color='#f38ba8', lw=2, linestyle='--')
    ax.plot(epochs_range, loss_ramanujan, label='Ramanujan-A (O Pulo Híbrido)', color='#89b4fa', lw=3)
    ax.plot(epochs_range, loss_adam_cosine, label='Adam Justo (Cosine + Warmup)', color='#a6e3a1', lw=3)
    
    ax.set_title('Laboratório Real: Desempenho no Dataset Hostil', color='#cdd6f4', fontsize=14)
    ax.set_xlabel('Épocas', color='#cdd6f4')
    ax.set_ylabel('Loss (BCE)', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    ax.grid(color='#313244', linestyle=':', alpha=0.5)
    ax.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    
    # Gráfico 2: Desmascarando a Alucinação (O segredo da Fração)
    ax = axs[1]
    ax.set_facecolor('#11111b')
    ax.plot(epochs_range, [0.01]*EPOCHS, label='LR Fixo do Adam Tradicional', color='#f38ba8', lw=2, linestyle='--')
    ax.plot(epochs_range, lr_ramanujan, label='Tamanho efetivo do Salto do Ramanujan-A', color='#89b4fa', lw=3)
    ax.plot(epochs_range, lr_cosine, label='LR Agendado (Cosine + Warmup Real)', color='#a6e3a1', lw=3)
    
    ax.set_title('O Truque: A Curva Orgânica da "Fração Contínua"', color='#cdd6f4', fontsize=14)
    ax.set_xlabel('Épocas', color='#cdd6f4')
    ax.set_ylabel('Tamanho do Passo (Learning Rate)', color='#cdd6f4')
    ax.tick_params(colors='#cdd6f4')
    ax.grid(color='#313244', linestyle=':', alpha=0.5)
    ax.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4')
    
    plt.tight_layout()
    plt.savefig('fair_benchmark_arena.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráficos da Verdade gerados: 'fair_benchmark_arena.png'")

if __name__ == "__main__":
    run_fair_benchmark()
