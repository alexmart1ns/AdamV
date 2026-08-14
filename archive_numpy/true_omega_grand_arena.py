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
# 1. ADAM JUSTO (O Campeão da Indústria)
# =========================================================
def adam_cosine_warmup(X, y, epochs=50, batch_size=32, max_lr=0.01, warmup_epochs=10, drift_epoch=None, true_W=None):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    
    # Copia para não alterar a referência global
    y_current = np.copy(y)
    
    for e in range(1, epochs + 1):
        if drift_epoch and e == drift_epoch:
            z = X @ (-true_W) + np.random.randn(N) * 0.5
            y_current = (z > 0).astype(float)
            
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
            X_b, y_b = X[idx], y_current[idx]
            
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
# 2. THE TRUE OMEGA (Engenharia de Ponta)
# =========================================================
def the_true_omega(X, y, epochs=50, batch_size=32, max_lr=0.02, weight_decay=0.001, drift_epoch=None, true_W=None):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    D = X.shape[1] 
    
    loss_ema = float('inf')
    patience = 0
    clock_reset_epoch = 0 
    
    y_current = np.copy(y)
    
    for e in range(1, epochs + 1):
        if drift_epoch and e == drift_epoch:
            z = X @ (-true_W) + np.random.randn(N) * 0.5
            y_current = (z > 0).astype(float)
            
        internal_e = e - clock_reset_epoch
        progresso = internal_e / epochs
        
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        
        for i in range(0, N, batch_size):
            t += 1
            idx = indices[i:i+batch_size]
            X_b, y_b = X[idx], y_current[idx]
            
            loss, grad = compute_loss_and_grad(X_b, y_b, W)
            epoch_loss += loss
            steps += 1
            
            m_v = 0.9 * m_v + 0.1 * grad
            v_v = 0.999 * v_v + 0.001 * (grad ** 2)
            m_hat = m_v / (1 - 0.9 ** t)
            v_hat = v_v / (1 - 0.999 ** t)
            
            direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # Envelope Ramanujan
            norm_dir_padrao = np.linalg.norm(direcao) / np.sqrt(D)
            envelope = 1.0 / (progresso + norm_dir_padrao + 1e-8)
            fator = 0.5 * (1 + math.cos(math.pi * progresso))
            lr_efetivo = max_lr * min(envelope * fator, 1.5)
            
            a = lr_efetivo * direcao
            
            # Freio Bakhshali Dinâmico
            explosao_mask = (np.abs(grad) > 3.0 * np.sqrt(v_hat))
            denom = np.abs(W) + np.abs(a) + 1e-8
            correction = (a ** 2) / (2.0 * denom)
            
            step_size = np.where(explosao_mask, a - np.sign(a) * correction, a)
            
            W = W - lr_efetivo * weight_decay * W
            W -= step_size
            
        avg_loss = epoch_loss / steps
        losses.append(avg_loss)
        
        # OMNI Basin Hopping (Estagnação / Concept Drift Hack)
        if e == 1:
            loss_ema = avg_loss
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * avg_loss
            
        if avg_loss > loss_ema * 0.95: # Sensibilidade alta para o teste de drift
            patience += 1
        else:
            patience = 0
            
        if patience >= 4 and clock_reset_epoch < e:
            # Ruído injetado
            W += np.random.randn(D) * 0.05 * np.std(W)
            patience = 0
            clock_reset_epoch = e
            
    return losses

def run_grand_arena_v2():
    print("==================================================================")
    print(" MEGA ARENA: THE TRUE OMEGA VS ADAM SOTA (4 CENÁRIOS EXTREMOS)")
    print("==================================================================")
    
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), facecolor='#1e1e2e')
    axs = axs.flatten()
    
    EPOCHS = 60
    BATCH_SIZE = 128
    
    # Parâmetros Base Comuns
    N, D = 4000, 300
    
    # ---------------------------------------------------------
    # CENÁRIO 1: O PADRÃO (Clean Dataset)
    # Prova que o Omega funciona bem em situações normais também.
    # ---------------------------------------------------------
    print("Gerando Cenário 1: Base Limpa...")
    np.random.seed(111)
    X1 = np.random.randn(N, D)
    W1 = np.random.randn(D) * 1.0
    y1 = ((X1 @ W1 + np.random.randn(N)*0.1) > 0).astype(float)
    
    loss_adam1 = adam_cosine_warmup(X1, y1, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.02)
    loss_omega1 = the_true_omega(X1, y1, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.03)
    
    axs[0].set_facecolor('#11111b')
    axs[0].plot(range(EPOCHS), loss_adam1, label='Adam SOTA', color='#f38ba8', lw=2, linestyle='--')
    axs[0].plot(range(EPOCHS), loss_omega1, label='True Omega', color='#a6e3a1', lw=3)
    axs[0].set_title('1. Padrão Ouro (Dataset Limpo)', color='#cdd6f4')
    axs[0].tick_params(colors='#cdd6f4')
    axs[0].grid(alpha=0.2)
    axs[0].legend()
    
    # ---------------------------------------------------------
    # CENÁRIO 2: ESPARSIDADE EXTREMA (NLP / Tabular Faltante)
    # ---------------------------------------------------------
    print("Gerando Cenário 2: Esparsidade Extrema (95% Zeros)...")
    np.random.seed(222)
    X2 = np.random.randn(N, D)
    X2[np.random.rand(N, D) > 0.05] = 0.0 # 95% zero
    W2 = np.random.randn(D) * 2.0
    y2 = ((X2 @ W2 + np.random.randn(N)*0.1) > 0).astype(float)
    
    loss_adam2 = adam_cosine_warmup(X2, y2, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.02)
    loss_omega2 = the_true_omega(X2, y2, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.03)
    
    axs[1].set_facecolor('#11111b')
    axs[1].plot(range(EPOCHS), loss_adam2, label='Adam SOTA', color='#f38ba8', lw=2, linestyle='--')
    axs[1].plot(range(EPOCHS), loss_omega2, label='True Omega', color='#a6e3a1', lw=3)
    axs[1].set_title('2. Deserto de Dados (95% Esparsidade)', color='#cdd6f4')
    axs[1].tick_params(colors='#cdd6f4')
    axs[1].grid(alpha=0.2)
    axs[1].legend()

    # ---------------------------------------------------------
    # CENÁRIO 3: BLACK SWAN (Outliers Colossais e Ruído)
    # ---------------------------------------------------------
    print("Gerando Cenário 3: Black Swan...")
    np.random.seed(333)
    X3 = np.random.randn(N, D)
    X3[np.random.rand(N, D) > 0.95] *= 1000.0 # Outliers explodindo
    W3 = np.random.randn(D) * 1.5
    z3 = X3 @ W3 + np.random.randn(N) * 2.0
    y3 = (z3 > 0).astype(float)
    flip_mask = np.random.rand(N) > 0.85
    y3[flip_mask] = 1.0 - y3[flip_mask] # 15% labels invertidas
    
    loss_adam3 = adam_cosine_warmup(X3, y3, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.01)
    loss_omega3 = the_true_omega(X3, y3, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.015)
    
    axs[2].set_facecolor('#11111b')
    axs[2].plot(range(EPOCHS), loss_adam3, label='Adam SOTA', color='#f38ba8', lw=2, linestyle='--')
    axs[2].plot(range(EPOCHS), loss_omega3, label='True Omega', color='#a6e3a1', lw=3)
    axs[2].set_title('3. Black Swan (Outliers Severos + Ruído)', color='#cdd6f4')
    axs[2].tick_params(colors='#cdd6f4')
    axs[2].grid(alpha=0.2)
    axs[2].legend()

    # ---------------------------------------------------------
    # CENÁRIO 4: CONCEPT DRIFT (As Regras do Universo Mudam)
    # ---------------------------------------------------------
    print("Gerando Cenário 4: Plasticidade Dinâmica (Drift)...")
    np.random.seed(444)
    X4 = np.random.randn(N, D)
    W4 = np.random.randn(D) * 1.5
    y4 = ((X4 @ W4 + np.random.randn(N)*0.5) > 0).astype(float)
    DRIFT_EPOCH = 25
    
    loss_adam4 = adam_cosine_warmup(X4, y4, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.02, drift_epoch=DRIFT_EPOCH, true_W=W4)
    loss_omega4 = the_true_omega(X4, y4, epochs=EPOCHS, batch_size=BATCH_SIZE, max_lr=0.03, drift_epoch=DRIFT_EPOCH, true_W=W4)
    
    print("\n[RESULTADOS FINAIS - LOSS BCE]")
    print(f"Cenário 1 (Limpo)       | Adam: {loss_adam1[-1]:.4f} | Omega: {loss_omega1[-1]:.4f}")
    print(f"Cenário 2 (Esparso)     | Adam: {loss_adam2[-1]:.4f} | Omega: {loss_omega2[-1]:.4f}")
    print(f"Cenário 3 (Black Swan)  | Adam: {loss_adam3[-1]:.4f} | Omega: {loss_omega3[-1]:.4f}")
    print(f"Cenário 4 (Drift)       | Adam: {loss_adam4[-1]:.4f} | Omega: {loss_omega4[-1]:.4f}")
    
    axs[3].set_facecolor('#11111b')
    axs[3].plot(range(EPOCHS), loss_adam4, label='Adam SOTA', color='#f38ba8', lw=2, linestyle='--')
    axs[3].plot(range(EPOCHS), loss_omega4, label='True Omega', color='#a6e3a1', lw=3)
    axs[3].axvline(x=DRIFT_EPOCH-1, color='#89b4fa', linestyle=':', label='Choque de Realidade')
    axs[3].set_title('4. Concept Drift (O Universo Inverte na Ep. 25)', color='#cdd6f4')
    axs[3].tick_params(colors='#cdd6f4')
    axs[3].grid(alpha=0.2)
    axs[3].legend()

    plt.tight_layout()
    plt.savefig('true_omega_grand_arena.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráficos da Mega Arena gerados com sucesso: 'true_omega_grand_arena.png'")

if __name__ == "__main__":
    run_grand_arena_v2()
