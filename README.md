# Documentação do GreedyDogSolver

Uma classe Python para resolver problemas de DOG (distribuição ótima de GPU's) usando algoritmos gulosos e métodos de melhoria iterativa baseado ma meta-heurística Iterated Greedy Algorithm.

## Como Executar

O programa pode ser executado de duas formas diferentes, usando o algoritmo guloso ou o otimizador Gurobi:

### Execução com Algoritmo Guloso

Usando o script `runner_greedydog.py`:

```bash
python runner_greedydog.py <output> <instance> [opções]
```

#### Argumentos Obrigatórios

- `output`: Caminho para o arquivo de saída onde a melhor solução será salva
- `instance`: Caminho para o arquivo de instância do problema

#### Opções

- `-i, --iterations`: Número máximo de iterações (padrão: 100000)
- `-t, --temperature`: Temperatura inicial para o solver (padrão: 0.3)
- `-s, --seed`: Semente para geração de números aleatórios (padrão: None)
- `-e, --enhanced`: Usar solução inicial aprimorada para melhorar o desempenho
- `-p, --plot`: Plotar distribuição da solução inicial e final

#### Exemplos de Uso

```bash
# Execução básica
python runner_greedydog.py output.csv instance.txt

# Execução com parâmetros personalizados
python runner_greedydog.py output.csv instance.txt -i 50000 -t 0.5 -s 42 -e -p
```

### Execução com Gurobi

Usando o script `runner_gurobi.py`:

```bash
python runner_gurobi.py <output> <instance> [opções]
```

#### Argumentos Obrigatórios

- `output`: Caminho para o arquivo de saída onde a melhor solução será salva
- `instance`: Caminho para o arquivo de instância do problema

#### Opções

- `-t, --time`: Limite de tempo para otimização em segundos (padrão: 100000)

#### Exemplos de Uso

```bash
# Execução básica
python runner_gurobi.py output.csv instance.txt

# Execução com limite de tempo personalizado (1 hora)
python runner_gurobi.py output.csv instance.txt -t 3600
```

## 

A classe `GreedyDogSolver` implementa algoritmos para atribuir PRNs (Nós de Processamento) a GPUs considerando restrições de VRAM e distribuição de tipos. O solucionador utiliza abordagens gulosas e de melhoria iterativa para encontrar alocações eficientes.

## Principais Recursos

- Carregamento de instâncias baseado em arquivo
- Múltiplas estratégias de solução:
  - Alocação gulosa básica
  - Solução inicial aprimorada com agrupamento por tipo
  - Melhoria iterativa com aceitação baseada em temperatura
  - Otimização com Gurobi (solução exata)
- Ferramentas de visualização e análise de soluções
- Saída CSV para soluções

## Métodos Principais

### Funcionalidades Principais

- `__init__(filename)`: Inicializa o solucionador com instância do problema a partir do arquivo
- `solve(...)`: Método principal de resolução com múltiplos parâmetros para personalização
- `initial_solution()`: Método de alocação gulosa básica
- `enhanced_initial_solution()`: Alocação inicial aprimorada com agrupamento por tipo
- `iterated_greedy(...)`: Método de melhoria iterativa com Iterated Greedy Algorithm
- `optimize_gurobi(...)`: Solução exata usando o solucionador Gurobi

### Métodos Auxiliares

- `mix_noloss()`: Combina GPUs sem exceder limites de VRAM
- `mix_loss()`: Une GPUs quando alguma perda de capacidade é aceitável
- `avaluate_solution()`: Calcula a qualidade da solução baseada na distribuição de tipos

### Análise e Saída

- `save_solution()`: Exporta solução para arquivo CSV
- `print_instance_info()`: Exibe detalhes da instância do problema
- `print_gpus_info()`: Mostra informações atuais de alocação de GPU
- `plot_distribution()`: Cria visualizações da solução atual

## Dependências

- gurobipy (para otimização exata)
- matplotlib (para visualização)
- csv (para entrada/saída de arquivo)
- random, time, math (para implementação do algoritmo)

