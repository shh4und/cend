# Guia de Versionamento Automático

Este documento explica as melhorias implementadas no sistema de versionamento automático de resultados.

## Mudanças Implementadas

### 1. **Versionamento Automático de Arquivos**

#### Arquivos SWC
Os arquivos de reconstrução agora são salvos com sufixos numéricos sequenciais:
- `OP_1_reconstruction01.swc`, `OP_1_reconstruction02.swc`, etc.
- `OP_2_reconstruction01.swc`, `OP_2_reconstruction02.swc`, etc.

#### Arquivos de Scores
Os arquivos CSV de scores também são versionados automaticamente:
- `scores01.csv`, `scores02.csv`, `scores03.csv`, etc.

### 2. **Metadata Atualizado**

O arquivo `.meta` agora usa:
- **`source_file`** ao invés de `source_image`
- Salva o **caminho completo** do arquivo ao invés de apenas o nome

### 3. **Reorganização do CSV**

O CSV de scores foi reorganizado com `SourceFile` como **última coluna** para acomodar nomes longos:

```csv
ImageID,DiademScore,FilterType,SigMin,SigMax,SigStep,NeuronThreshold,PruningThreshold,GreyStrElSize,GreyStrElWeight,SourceFile
```

## Como Usar

### Pipeline de Reconstrução

#### Auto-detecção de Versão (Recomendado)
```bash
python -m cend.processing.pipeline
```
O sistema automaticamente detecta a próxima versão disponível (01, 02, 03...).

#### Especificar Versão Manualmente
```bash
python -m cend.processing.pipeline --output_suffix 01
python -m cend.processing.pipeline --output_suffix 02
```

### Avaliação com Make

```bash
make evaluate
```

O makefile automaticamente:
1. Detecta a próxima versão de `scores.csv` disponível
2. Procura pelos arquivos SWC correspondentes com o mesmo sufixo
3. Cria o arquivo `scoresXX.csv` com os resultados

### Limpeza

```bash
make clean
```

Remove todos os arquivos `scores*.csv` do diretório `./scores/`.

## Exemplos de Workflow

### Teste 1 - Parâmetros Padrão
```bash
# Executar pipeline (cria arquivos *01.swc)
python -m cend.processing.pipeline

# Avaliar resultados (cria scores01.csv)
make evaluate
```

### Teste 2 - Ajustar Sigma
```bash
# Executar com novos parâmetros (cria arquivos *02.swc)
python -m cend.processing.pipeline --sigma_min 1.5 --sigma_max 3.0

# Avaliar novos resultados (cria scores02.csv)
make evaluate
```

### Teste 3 - Filtro Diferente
```bash
# Testar outro filtro (cria arquivos *03.swc)
python -m cend.processing.pipeline --filter_type frangi --output_suffix 03

# Avaliar (cria scores03.csv)
make evaluate
```

## Comparação de Resultados

Agora você pode manter múltiplas versões e comparar facilmente:

```bash
# Ver todos os resultados
ls scores/scores*.csv
ls results_swc/OP_*_reconstruction*.swc

# Comparar scores
column -s, -t scores/scores01.csv
column -s, -t scores/scores02.csv
column -s, -t scores/scores03.csv
```

## Vantagens

✅ **Sem sobreposição**: Cada execução gera arquivos únicos  
✅ **Rastreabilidade**: Todos os parâmetros são salvos no `.meta`  
✅ **Comparação fácil**: Múltiplas versões lado a lado  
✅ **Automático**: Não precisa gerenciar numeração manualmente  
✅ **Seguro**: Nunca sobrescreve resultados anteriores  

## Estrutura de Arquivos

```
results_swc/
├── OP_1_reconstruction01.swc
├── OP_1_reconstruction01.meta
├── OP_1_reconstruction02.swc
├── OP_1_reconstruction02.meta
├── OP_2_reconstruction01.swc
├── OP_2_reconstruction01.meta
└── ...

scores/
├── scores01.csv
├── scores02.csv
└── scores03.csv
```

## Notas Técnicas

- O sufixo é sempre formatado com 2 dígitos (`01`, `02`, ..., `99`)
- A auto-detecção verifica apenas o arquivo `OP_1_reconstruction*.swc`
- O makefile sincroniza automaticamente a versão entre SWC e CSV
- Todos os metadados continuam sendo salvos nos arquivos `.meta`
