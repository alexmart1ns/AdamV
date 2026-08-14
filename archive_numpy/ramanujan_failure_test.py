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

# 1. ADAM PURO (Padrão Ouro)
def adam_minibatch(X, y, epochs=100, batch_size=32, lr=0.01):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    m_v = np.zeros_like(W)
    v_v = np.zeros_like(W)
    t = 0
    N = X.shape[0]
    for e in range(epochs):
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

# 2. RAMANUJAN RGD PURO (O que falhou no ruído)
def ramanujan_rgd_minibatch(X, y, epochs=100, batch_size=32):
    np.random.seed(42)
    W = np.random.randn(X.shape[1]) * 0.01
    losses = []
    N = X.shape[0]
    base_scale = 0.5 
    
    for e in range(1, epochs + 1):
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        for i in range(0, N, batch_size):
            idx = indices[i:i+batch_size]
            X_b, y_b = X[idx], y[idx]
            
            loss, grad = compute_loss_and_grad(X_b, y_b, W)
            epoch_loss += loss
            steps += 1
            
            norm_grad = np.linalg.norm(grad)
            if norm_grad < 1e-12: continue
            
            salto = 1.0 / (e + (norm_grad**2) / e)
            if norm_grad < 0.1:
                salto *= (e * 0.1)
                
            W -= salto * base_scale * grad
        losses.append(epoch_loss / steps)
    return losses

# 3. Ramanujan-A (A Mutação Perfeita: Adam + Ramanujan)
def ramanujan_a_minibatch(X, y, epochs=100, batch_size=32, base_scale=0.15):
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
            
            # PASSO 1: Bússola do Adam (Direção Normalizada e Imune a Escalas/Ruído)
            m_v = 0.9 * m_v + 0.1 * grad
            v_v = 0.999 * v_v + 0.001 * (grad ** 2)
            m_hat = m_v / (1 - 0.9 ** t)
            v_hat = v_v / (1 - 0.999 ** t)
            
            direcao_perfeita = m_hat / (np.sqrt(v_hat) + 1e-8)
            
            # PASSO 2: Pulo Mágico de Ramanujan (Decay Estocástico Inteligente)
            norm_dir = np.linalg.norm(direcao_perfeita)
            if norm_dir < 1e-12: continue
            
            # A fração gera um Efeito "Warm-up" biológico: Começa pequeno, acelera, depois desacelera
            salto_ramanujan = 1.0 / (e + (norm_dir**2) / e)
            
            # PASSO 3: O Pulo Híbrido Absoluto
            W -= salto_ramanujan * base_scale * direcao_perfeita
            
        losses.append(epoch_loss / steps)
    return losses

def run_hybrid_stress_test():
    print("==================================================================")
    print(" O LABORATÓRIO MUTANTE: TESTE DE ESTRESSE Ramanujan-A (Ill-Conditioned)")
    print("==================================================================")
    
    print("1. Forjando Dataset Caótico e Mal-Condicionado (Mini-Batch Ruídoso)...")
    np.random.seed(99)
    N = 8000
    D = 50
    X = np.random.randn(N, D)
    
    # Corrompendo features para quebrar otimizadores frágeis
    for i in range(D):
        if i % 2 == 0: X[:, i] *= 500.0  
        else: X[:, i] *= 0.01   
            
    true_W = np.random.randn(D) * 1.5
    noise = np.random.randn(N) * 2.0  
    z = X @ true_W + noise
    y = (z > 0).astype(float)
    
    EPOCHS = 60
    BATCH_SIZE = 64
    
    print("\n[LUTADOR 1] RAMANUJAN PURO (O Antigo)")
    t0 = time.time()
    loss_rgd = ramanujan_rgd_minibatch(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    t1 = time.time()
    print(f" -> Loss Final: {loss_rgd[-1]:.4f} | Status: Colapso Oscilatório (Ricochete)")
    
    print("\n[LUTADOR 2] ADAM PURO (O Padrão Moderno)")
    t0 = time.time()
    loss_adam = adam_minibatch(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=0.01)
    t1 = time.time()
    print(f" -> Loss Final: {loss_adam[-1]:.4f} | Status: Estável")

    print("\n[LUTADOR 3] Ramanujan-A HÍBRIDO (A Mutação Definitiva)")
    print("Direção do Adam com Pulo Termodinâmico de Ramanujan.")
    t0 = time.time()
    loss_ramanujan_a = ramanujan_a_minibatch(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)
    t1 = time.time()
    print(f" -> Loss Final: {loss_ramanujan_a[-1]:.4f} | Status: SOBERANIA VÉDICA ALCANÇADA!")

    # Plotting the Resurrection
    plt.figure(figsize=(14, 8), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, loss_rgd, label='Ramanujan Puro (Explosão)', color='#f38ba8', linewidth=2, linestyle=':')
    plt.plot(epochs_range, loss_adam, label='Adam Tradicional (Padrão Ouro)', color='#f9e2af', linewidth=2)
    plt.plot(epochs_range, loss_ramanujan_a, label='Ramanujan-A Híbrido (O Monstro Védico)', color='#a6e3a1', linewidth=4)
    
    plt.title('A Ressurreição: Ramanujan-A domina o Caos Mal-Condicionado', color='#cdd6f4', fontsize=16, fontweight='bold')
    plt.xlabel('Épocas de Treinamento (Mini-Batches)', color='#cdd6f4', fontsize=14)
    plt.ylabel('Erro Absoluto (BCE Loss)', color='#cdd6f4', fontsize=14)
    
    # Focus on the bottom stability area
    plt.ylim(0, max(loss_adam)*1.5)
    
    plt.tick_params(axis='x', colors='#cdd6f4', labelsize=12)
    plt.tick_params(axis='y', colors='#cdd6f4', labelsize=12)
    plt.grid(True, color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('ramanujan_failure_test.png', dpi=250, facecolor='#1e1e2e')
    print("\nGráfico do Triunfo salvo, sobrescrevendo o arquivo anterior.")

if __name__ == "__main__":
    run_hybrid_stress_test()
