import numpy as np
import matplotlib.pyplot as plt
import time

def sigmoid(z):
    z = np.clip(z, -250, 250)
    return 1.0 / (1.0 + np.exp(-z))

def compute_loss_and_grad(X, y, W):
    m = X.shape[0]
    preds = sigmoid(X @ W)
    loss = -np.mean(y * np.log(preds + 1e-15) + (1 - y) * np.log(1 - preds + 1e-15))
    grad = (X.T @ (preds - y)) / m
    return loss, grad

# ADAM PURO
def adam_minibatch(X, y, epochs=50, batch_size=32, lr=0.01, drift_epoch=None, true_W=None):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    for e in range(1, epochs + 1):
        # Para o teste de Concept Drift (Mundo muda no meio do treino)
        if drift_epoch and e == drift_epoch:
            z = X @ (-true_W) + np.random.randn(N) * 0.5
            y = (z > 0).astype(float)
            
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

# RAMANUJAN-A HÍBRIDO
def ramanujan_a_minibatch(X, y, epochs=50, batch_size=32, base_scale=0.15, drift_epoch=None, true_W=None):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    
    for e in range(1, epochs + 1):
        if drift_epoch and e == drift_epoch:
            z = X @ (-true_W) + np.random.randn(N) * 0.5
            y = (z > 0).astype(float)
            
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
            norm_dir = np.linalg.norm(direcao)
            if norm_dir < 1e-12: continue
            
            salto_ramanujan = 1.0 / (e + (norm_dir**2) / e)
            W -= salto_ramanujan * base_scale * direcao
            
        losses.append(epoch_loss / steps)
    return losses

def run_grand_arena():
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), facecolor='#1e1e2e')
    EPOCHS = 40
    
    # ---------------------------------------------------------
    # CENÁRIO 1: ESPARSIDADE EXTREMA (Simulando NLP / Texto)
    # ---------------------------------------------------------
    print("Testando Cenário 1: Esparsidade (90% dos dados são zeros)...")
    np.random.seed(101)
    N1, D1 = 3000, 200
    X1 = np.random.randn(N1, D1)
    # Mascara de Esparsidade (90% zero)
    mask = np.random.rand(N1, D1) > 0.9
    X1 = X1 * mask
    W1 = np.random.randn(D1)
    y1 = ((X1 @ W1 + np.random.randn(N1)*0.1) > 0).astype(float)
    
    loss_adam_1 = adam_minibatch(X1, y1, epochs=EPOCHS, lr=0.02)
    loss_ramanujana_1 = ramanujan_a_minibatch(X1, y1, epochs=EPOCHS, base_scale=0.2)
    
    axs[0].set_facecolor('#11111b')
    axs[0].plot(range(EPOCHS), loss_adam_1, label='Adam', color='#f9e2af', lw=2)
    axs[0].plot(range(EPOCHS), loss_ramanujana_1, label='Ramanujan-A', color='#a6e3a1', lw=3)
    axs[0].set_title('1. Esparsidade (NLP / Texto)', color='#cdd6f4')
    axs[0].set_ylabel('Loss', color='#cdd6f4')
    axs[0].tick_params(colors='#cdd6f4')
    axs[0].grid(alpha=0.2)
    axs[0].legend()
    
    # ---------------------------------------------------------
    # CENÁRIO 2: BLACK SWAN (Outliers Extremos e Rótulos Trocados)
    # ---------------------------------------------------------
    print("Testando Cenário 2: Black Swan (Outliers e Ruído)...")
    np.random.seed(102)
    N2, D2 = 3000, 50
    X2 = np.random.randn(N2, D2)
    
    # Injetando Outliers Brutais (5% dos dados explodem x1000)
    outlier_mask = np.random.rand(N2, D2) > 0.95
    X2[outlier_mask] *= 1000.0
    
    W2 = np.random.randn(D2)
    z2 = X2 @ W2 + np.random.randn(N2)
    y2 = (z2 > 0).astype(float)
    # Corrompendo 10% das respostas (Ruído de Label)
    flip_mask = np.random.rand(N2) > 0.90
    y2[flip_mask] = 1.0 - y2[flip_mask]
    
    loss_adam_2 = adam_minibatch(X2, y2, epochs=EPOCHS, lr=0.01)
    loss_ramanujana_2 = ramanujan_a_minibatch(X2, y2, epochs=EPOCHS, base_scale=0.1)
    
    axs[1].set_facecolor('#11111b')
    axs[1].plot(range(EPOCHS), loss_adam_2, label='Adam', color='#f9e2af', lw=2)
    axs[1].plot(range(EPOCHS), loss_ramanujana_2, label='Ramanujan-A', color='#a6e3a1', lw=3)
    axs[1].set_title('2. Cisne Negro (Outliers/Ruído)', color='#cdd6f4')
    axs[1].tick_params(colors='#cdd6f4')
    axs[1].grid(alpha=0.2)
    axs[1].legend()

    # ---------------------------------------------------------
    # CENÁRIO 3: CONCEPT DRIFT (As Leis da Física mudam na Época 20)
    # ---------------------------------------------------------
    print("Testando Cenário 3: Concept Drift (O Universo Vira de Cabeça Para Baixo)...")
    np.random.seed(103)
    N3, D3 = 3000, 50
    X3 = np.random.randn(N3, D3)
    W3 = np.random.randn(D3) * 2.0
    y3 = ((X3 @ W3 + np.random.randn(N3)*0.5) > 0).astype(float)
    
    # Na época 20, y3 vai ser recalculado com o peso INVERTIDO (-W3)
    DRIFT_EPOCH = 20
    
    loss_adam_3 = adam_minibatch(X3, y3, epochs=EPOCHS, lr=0.01, drift_epoch=DRIFT_EPOCH, true_W=W3)
    loss_ramanujana_3 = ramanujan_a_minibatch(X3, y3, epochs=EPOCHS, base_scale=0.15, drift_epoch=DRIFT_EPOCH, true_W=W3)
    
    axs[2].set_facecolor('#11111b')
    axs[2].plot(range(EPOCHS), loss_adam_3, label='Adam', color='#f9e2af', lw=2)
    axs[2].plot(range(EPOCHS), loss_ramanujana_3, label='Ramanujan-A', color='#a6e3a1', lw=3)
    
    axs[2].axvline(x=DRIFT_EPOCH-1, color='#f38ba8', linestyle='--', label='Choque (Física Invertida)')
    
    axs[2].set_title('3. Plasticidade Dinâmica (Choque de Realidade)', color='#cdd6f4')
    axs[2].tick_params(colors='#cdd6f4')
    axs[2].grid(alpha=0.2)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('ramanujan_a_grand_arena.png', dpi=200, facecolor='#1e1e2e')
    print("Gráficos gerados com sucesso na Grande Arena!")

if __name__ == "__main__":
    run_grand_arena()
