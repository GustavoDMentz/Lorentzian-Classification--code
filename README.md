# Lorentzian Classification — Estratégia de Trading

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-kNN-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![TA-Lib](https://img.shields.io/badge/TA--Lib-Indicadores-4CAF50?style=flat-square)](https://ta-lib.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Implementação em Python da estratégia **Lorentzian Classification**, originalmente desenvolvida por [jdehorty](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/) no TradingView (PineScript). Usa **K-Nearest Neighbors com distância de Lorentz** para classificar condições de mercado e gerar sinais de compra e venda.

---

## 📖 O que é a Classificação Lorentziana?

A estratégia Lorentzian Classification é um algoritmo de machine learning supervisionado que usa **kNN (K-Nearest Neighbors)** com uma métrica de distância não-euclidiana — a **distância Lorentziana** — para prever a direção futura do preço com base em padrões históricos de indicadores técnicos.

### Por que distância Lorentziana?

A distância euclidiana tradicional trata todos os desvios de forma quadrática, o que a torna sensível a outliers. A distância Lorentziana usa escala logarítmica:

$$d_L(x, y) = \sum_{i=1}^{n} \ln(1 + |x_i - y_i|)$$

Isso resulta em:
- **Maior robustez** a movimentos extremos de mercado
- **Melhor separação** entre padrões similares e dissimilares
- **Menos sensibilidade** a spikes de volatilidade

---

## 🔬 Arquitetura da estratégia

### Features (Indicadores de Entrada)

O modelo usa 5 indicadores técnicos normalizados como features do kNN:

| Feature | Indicador | Parâmetros | Range Normalizado |
|---------|-----------|------------|-------------------|
| f1 | RSI | período 14 | [-1, 1] |
| f2 | Wave Trend (WT) | canal 10, média 11 | [-1, 1] |
| f3 | CCI | período 20 | normalizado / 100 |
| f4 | ADX | período 20 | [-1, 1] |
| f5 | RSI | período 9 | [-1, 1] |

### Processo de Classificação

```
Para cada barra atual:
  1. Calcular features [f1..f5] normalizadas
  2. Buscar os K vizinhos mais próximos no histórico (distância Lorentziana)
  3. Classificar por maioria de votos: Long (1), Short (-1), Neutro (0)
  4. Aplicar filtros de mercado
  5. Gerar sinal de entrada/saída
```

### Rótulos de Treinamento (Auto-supervisionado)

Os rótulos são gerados automaticamente com base no movimento de preço 4 barras à frente:

```python
price_change = (close[i+4] - close[i]) / close[i]

if price_change > +0.1% → Long  (+1)
if price_change < -0.1% → Short (-1)
else                     → Neutro (0)
```

---

## 🛡️ Filtros de Mercado

A estratégia inclui 3 filtros opcionais que suprimem sinais em condições desfavoráveis:

| Filtro | Descrição | Ativado por padrão |
|--------|-----------|-------------------|
| **Volatility Filter** | Só opera quando volatilidade está acima do percentil 10 histórico | ✅ |
| **Regime Filter** | Detecta tendência vs lateralização via média de retornos (50 barras) | ✅ |
| **ADX Filter** | Exige ADX > threshold para operar (mercados com tendência definida) | ❌ |

---

## 🚀 Instalação

### Pré-requisitos

> ⚠️ **TA-Lib** requer instalação da biblioteca C antes do pacote Python.

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# Mac
brew install ta-lib
```

**Windows:**  
Baixe o instalador em: https://sourceforge.net/projects/ta-lib/files/ta-lib/

### Instale as dependências Python

```bash
pip install pandas numpy scikit-learn TA-Lib
```

---

## ▶️ Como usar

### Teste com dados sintéticos (exemplo incluído)

```bash
python "Código Principal"
```

O script irá gerar 1.000 barras sintéticas e exibir:

```
Testando a estratégia Lorentzian Classification...

Estatísticas dos Sinais:
Total de sinais: 47
Sinais de compra: 24
Sinais de venda: 23

Últimos sinais gerados:
              close  signal  prediction
2024-01-15    102.3       1           1
2024-01-18    101.1      -1          -1
...
```

### Uso com dados reais (OHLCV)

```python
import pandas as pd
from estrategia import LorentzianClassificationStrategy

# Carregue seus dados OHLCV
# Colunas obrigatórias: open, high, low, close, volume
df = pd.read_csv('seus_dados.csv', index_col=0, parse_dates=True)

# Instanciar a estratégia
strategy = LorentzianClassificationStrategy(
    neighbors_count=8,       # Número de vizinhos kNN
    max_bars_back=2000,      # Máximo de barras históricas buscadas
    feature_count=5,         # Número de features usadas
    use_volatility_filter=True,
    use_regime_filter=True,
    use_adx_filter=False,
    regime_threshold=-0.1,
    adx_threshold=20
)

# Gerar sinais
results = strategy.generate_signals(df)

# Visualizar
print(results[results['signal'] != 0][['close', 'signal', 'prediction']])
```

### Retorno de `generate_signals()`

O DataFrame retornado contém:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `close` | float | Preço de fechamento |
| `prediction` | int | Predição kNN: 1 (long), -1 (short), 0 (neutro) |
| `filter_passed` | bool | Se os filtros de mercado foram aprovados |
| `signal` | int | Sinal gerado: 1 (compra/fecha short), -1 (venda/fecha long), 0 |
| `position` | int | Posição atual: 1 (long), -1 (short), 0 (flat) |

---

## ⚙️ Parâmetros configuráveis

```python
LorentzianClassificationStrategy(
    neighbors_count=8,         # K do kNN — mais vizinhos = mais suavizado mas mais lento
    max_bars_back=2000,        # Janela de busca histórica — reduzir para menor custo computacional
    feature_count=5,           # Número de features ativas (1-5)
    use_volatility_filter=True, # Filtra mercados sem volatilidade
    use_regime_filter=True,    # Filtra mercados sem tendência
    use_adx_filter=False,      # Filtro ADX adicional (mais restritivo)
    regime_threshold=-0.1,     # Threshold do filtro de regime (negativo = permissivo)
    adx_threshold=20           # ADX mínimo para o filtro ADX
)
```

### Configuração de features

Você pode alterar os indicadores usados modificando `feature_config`:

```python
strategy.feature_config = {
    'f1': {'type': 'RSI', 'param_a': 14, 'param_b': 1},   # RSI(14)
    'f2': {'type': 'WT',  'param_a': 10, 'param_b': 11},  # WaveTrend(10, 11)
    'f3': {'type': 'CCI', 'param_a': 20, 'param_b': 1},   # CCI(20)
    'f4': {'type': 'ADX', 'param_a': 20, 'param_b': 2},   # ADX(20)
    'f5': {'type': 'RSI', 'param_a': 9,  'param_b': 1},   # RSI(9)
}
```

Tipos disponíveis: `RSI`, `WT` (Wave Trend), `CCI`, `ADX`

---

## ⚠️ Avisos importantes

> **Este código é para fins educacionais e de pesquisa.** Não constitui conselho financeiro ou recomendação de investimento.

- **Backtesting ≠ resultados futuros**: Qualquer estratégia pode ter bom desempenho histórico e falhar no mercado ao vivo.
- **Overfitting**: O kNN com lookback longo pode se ajustar demais aos dados históricos.
- **Latência**: Com `max_bars_back=2000`, o cálculo pode ser lento para datasets grandes — considere reduzir para dados de alta frequência.
- **Slippage e custos**: O backtest não inclui spread, comissão ou slippage — adicione ao avaliar resultados.

---

## 🧠 Referência original

Baseado no indicador PineScript:  
**[Machine Learning: Lorentzian Classification](https://www.tradingview.com/script/WhBzgfDu-Machine-Learning-Lorentzian-Classification/)** — por *jdehorty* no TradingView.

---

## 📄 Licença

MIT — uso livre para fins educacionais e de pesquisa.
