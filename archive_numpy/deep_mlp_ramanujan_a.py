import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. FUNÇÕES DE ATIVAÇÃO E LOSS
# ==========================================
def relu(Z):
    return np.maximum(0, Z)

def sigmoid(Z):
    Z = np.clip(Z, -250, 250)
    return 1.0 / (1.0 + np.exp(-Z))

def bce_loss(Y_hat, Y):
    return -np.mean(Y * np.log(Y_hat + 1e-15) + (1 - Y) * np.log(1 - Y_hat + 1e-15))

# ==========================================
# 2. MOTOR DE REDES NEURAIS (DEEP MLP)
# ==========================================
def init_params(input_dim, h1_dim, h2_dim, out_dim=1):
    np.random.seed(99)
    # Inicialização He (Ideal para ReLU)
    return {
        'W1': np.random.randn(input_dim, h1_dim) * np.sqrt(2.0/input_dim),
        'b1': np.zeros((1, h1_dim)),
        'W2': np.random.randn(h1_dim, h2_dim) * np.sqrt(2.0/h1_dim),
        'b2': np.zeros((1, h2_dim)),
        'W3': np.random.randn(h2_dim, out_dim) * np.sqrt(2.0/h2_dim),
        'b3': np.zeros((1, out_dim))
    }

def forward_pass(X, params):
    cache = {}
    cache['Z1'] = X @ params['W1'] + params['b1']
    cache['A1'] = relu(cache['Z1'])
    
    cache['Z2'] = cache['A1'] @ params['W2'] + params['b2']
    cache['A2'] = relu(cache['Z2'])
    
    cache['Z3'] = cache['A2'] @ params['W3'] + params['b3']
    cache['A3'] = sigmoid(cache['Z3'])
    return cache['A3'], cache

def backward_pass(X, Y, cache, params):
    m = X.shape[0]
    grads = {}
    
    # Chain Rule Camada 3 (Output)
    dZ3 = cache['A3'] - Y
    grads['W3'] = (cache['A2'].T @ dZ3) / m
    grads['b3'] = np.sum(dZ3, axis=0, keepdims=True) / m
    
    # Chain Rule Camada 2 (Hidden)
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * (cache['Z2'] > 0).astype(float)
    grads['W2'] = (cache['A1'].T @ dZ2) / m
    grads['b2'] = np.sum(dZ2, axis=0, keepdims=True) / m
    
    # Chain Rule Camada 1 (Input Hidden)
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * (cache['Z1'] > 0).astype(float)
    grads['W1'] = (X.T @ dZ1) / m
    grads['b1'] = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return grads

# ==========================================
# 3. OTIMIZADORES PROFUNDOS
# ==========================================
def train_deep_model(X_train, Y_train, X_val, Y_val, optimizer_type, epochs=300, batch_size=64, lr=0.01):
    params = init_params(X_train.shape[1], 64, 32, 1)
    
    # Estados de Memória do Adam (Dicionários para cada matriz W e b)
    m_v = {k: np.zeros_like(v) for k, v in params.items()}
    v_v = {k: np.zeros_like(v) for k, v in params.items()}
    
    train_losses = []
    val_losses = []
    
    N = X_train.shape[0]
    t = 0
    
    for e in range(1, epochs + 1):
        indices = np.random.permutation(N)
        epoch_loss = 0
        steps = 0
        
        for i in range(0, N, batch_size):
            t += 1
            idx = indices[i:i+batch_size]
            X_b, Y_b = X_train[idx], Y_train[idx]
            
            # Forward e Backward Clássico
            A3, cache = forward_pass(X_b, params)
            batch_loss = bce_loss(A3, Y_b)
            epoch_loss += batch_loss
            steps += 1
            
            grads = backward_pass(X_b, Y_b, cache, params)
            
            # Atualização dos Pesos O(N)
            for key in params.keys():
                grad = grads[key]
                
                # Bússola do Adam
                m_v[key] = 0.9 * m_v[key] + 0.1 * grad
                v_v[key] = 0.999 * v_v[key] + 0.001 * (grad ** 2)
                m_hat = m_v[key] / (1 - 0.9 ** t)
                v_hat = v_v[key] / (1 - 0.999 ** t)
                
                direcao = m_hat / (np.sqrt(v_hat) + 1e-8)
                
                if optimizer_type == 'adam':
                    # Passo Estático (O Padrão da Indústria)
                    params[key] -= lr * direcao
                    
                elif optimizer_type == 'ramanujan_a':
                    # O Pulo Termodinâmico Híbrido (O Segredo Milenar)
                    norm_dir = np.linalg.norm(direcao)
                    if norm_dir > 1e-12:
                        salto = 1.0 / (e + (norm_dir**2) / e)
                        params[key] -= salto * (lr * 10) * direcao  # scale comp factor
        
        # Salvando Loss de Treino e Validação da Época
        train_losses.append(epoch_loss / steps)
        
        # Teste de Generalização (A Prova Real) no Dataset de Validação
        A3_val, _ = forward_pass(X_val, params)
        val_losses.append(bce_loss(A3_val, Y_val))
        
    return train_losses, val_losses

# ==========================================
# 4. GERAÇÃO DE DADOS NÃO-LINEARES E ARENA
# ==========================================
def run_deep_learning_arena():
    print("==========================================================")
    print(" O BATISMO DE FOGO: DEEP MLP BACKPROPAGATION & OVERFITTING")
    print("==========================================================")
    
    print("1. Forjando Espiral Caótica (Dados Não-Lineares Severos)...")
    np.random.seed(42)
    N = 8000
    # Duas espirais aninhadas com alto ruído
    theta = np.sqrt(np.random.rand(N)) * 3 * np.pi
    r = 2 * theta + np.pi
    X_a = np.c_[r * np.cos(theta), r * np.sin(theta)] + np.random.randn(N, 2) * 1.5
    X_b = np.c_[-r * np.cos(theta), -r * np.sin(theta)] + np.random.randn(N, 2) * 1.5
    
    X_full = np.vstack([X_a, X_b])
    Y_full = np.hstack([np.zeros(N), np.ones(N)]).reshape(-1, 1)
    
    # Embaralhando e Dividindo (80% Treino / 20% Validação)
    indices = np.random.permutation(X_full.shape[0])
    train_idx, val_idx = indices[:int(0.8*2*N)], indices[int(0.8*2*N):]
    X_train, Y_train = X_full[train_idx], Y_full[train_idx]
    X_val, Y_val = X_full[val_idx], Y_full[val_idx]
    
    EPOCHS = 150
    print(f"Dataset Dividido: {X_train.shape[0]} Treino, {X_val.shape[0]} Validação.")
    print(f"Arquitetura: 2 -> 64 (ReLU) -> 32 (ReLU) -> 1 (Sigmoid)")
    
    print("\n[Treinando Rede Neural Oculta com ADAM PURO]")
    t0 = time.time()
    adam_train, adam_val = train_deep_model(X_train, Y_train, X_val, Y_val, 'adam', epochs=EPOCHS, lr=0.005)
    print(f" -> Concluído em {time.time()-t0:.2f}s | Val Loss Final: {adam_val[-1]:.4f}")
    
    print("\n[Treinando Rede Neural Oculta com Ramanujan-A HÍBRIDO]")
    t0 = time.time()
    ramanujan_a_train, ramanujan_a_val = train_deep_model(X_train, Y_train, X_val, Y_val, 'ramanujan_a', epochs=EPOCHS, lr=0.005)
    print(f" -> Concluído em {time.time()-t0:.2f}s | Val Loss Final: {ramanujan_a_val[-1]:.4f}")
    
    # Plotting
    plt.figure(figsize=(14, 8), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#11111b')
    
    epochs_range = range(1, EPOCHS + 1)
    
    # Adam (Tracejado = Treino, Solido = Validacao)
    plt.plot(epochs_range, adam_train, label='Adam (Treino)', color='#f9e2af', lw=1, linestyle='--')
    plt.plot(epochs_range, adam_val, label='Adam (Validação - Mundo Real)', color='#f9e2af', lw=3)
    
    # Ramanujan-A
    plt.plot(epochs_range, ramanujan_a_train, label='Ramanujan-A (Treino)', color='#a6e3a1', lw=1, linestyle='--')
    plt.plot(epochs_range, ramanujan_a_val, label='Ramanujan-A (Validação - Mundo Real)', color='#a6e3a1', lw=3)
    
    plt.title('O Gap de Generalização: Treinamento Profundo (Deep MLP) em Dados Espirais', color='#cdd6f4', fontsize=16)
    plt.xlabel('Épocas (Backpropagation)', color='#cdd6f4', fontsize=14)
    plt.ylabel('Erro (Binary Cross-Entropy)', color='#cdd6f4', fontsize=14)
    
    plt.ylim(0, max(adam_train)*1.1)
    
    plt.tick_params(axis='x', colors='#cdd6f4', labelsize=12)
    plt.tick_params(axis='y', colors='#cdd6f4', labelsize=12)
    plt.grid(True, color='#313244', linestyle=':', alpha=0.5)
    plt.legend(facecolor='#181825', edgecolor='#313244', labelcolor='#cdd6f4', fontsize=13)
    
    plt.tight_layout()
    plt.savefig('deep_learning_benchmark.png', dpi=250, facecolor='#1e1e2e')
    print("\n[SUCESSO] Gráfico Final de Deep Learning salvo: deep_learning_benchmark.png")

if __name__ == "__main__":
    run_deep_learning_arena()
