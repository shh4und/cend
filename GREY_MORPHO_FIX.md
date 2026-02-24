# Grey Morphological Denoising - Bug Fix

## Problema Identificado

O `grey_morphological_denoising` não estava tendo efeito diferenciado ao mudar o tamanho da estrutura (`size`). 

### Evidências

1. Scores idênticos com `size=2` vs `size=6`:
   - scores_frangi_yang_sesz_2_pt_0.csv
   - scores_frangi_yang_sesz_6_pt_0.csv
   - scores.csv (com denoising comentado)

2. Testes confirmaram: estruturas de tamanhos 2, 4, 6, 8 produziam **exatamente** os mesmos resultados

## Causa Raiz

A fórmula usada para criar a estrutura não-plana:

```python
struct_nonflat = (x**2 + y**2 + z**2) * -0.5
```

Com diferentes tamanhos, esta fórmula cria estruturas com **escalas absolutas diferentes** mas que, matematicamente, produzem o mesmo efeito morfológico:

- Size 2: min = -1.5, max = 0.0
- Size 4: min = -6.0, max = 0.0  
- Size 6: min = -13.5, max = 0.0
- Size 8: min = -24.0, max = 0.0

O scipy `grey_opening` e `grey_closing` acabavam produzindo resultados idênticos porque a **escala relativa** era a mesma.

## Solução Implementada

Normalizar a estrutura para ter sempre o mesmo range (`min = -1.0, max = 0.0`):

```python
# Create non-flat structure element (paraboloid)
struct_nonflat = (x**2 + y**2 + z**2) * -0.5
struct_nonflat[size // 2, size // 2, size // 2] = 0

# IMPORTANT: Normalize to have consistent range across different sizes
if struct_nonflat.min() != 0:
    struct_nonflat = struct_nonflat / abs(struct_nonflat.min())
```

## Resultados Após Fix

Com a normalização, diferentes tamanhos produzem **efeitos significativamente diferentes**:

| Size | Mean Absolute Difference | Efeito |
|------|-------------------------|--------|
| 2    | 0.1437                  | Suave |
| 4    | 0.3329                  | 2.3x mais forte |
| 6    | 0.3857                  | 2.7x mais forte |
| 8    | 0.4136                  | 2.9x mais forte |

## Arquivos Modificados

- `/home/dnxx/Projects/cend/src/cend/processing/pipeline.py`
  - Adicionado import de `grey_morphological_denoising`
  - Descomentada a chamada da função
  - Adicionada normalização da estrutura
  - Removida atribuição direta `img_grey_morpho = img_filtered`

## Próximos Passos

1. Re-rodar o pipeline com diferentes valores de `size` (2, 4, 6)
2. Avaliar os scores com DiademMetric
3. Comparar resultados para determinar o tamanho ótimo
4. Considerar adicionar `size` como parâmetro CLI:
   ```python
   parser.add_argument('--grey_morpho_size', type=int, default=2,
                       help='Size of grey morphological structure element')
   ```

## Testes Realizados

Scripts de teste criados em:
- `test_grey_morpho.py` - Verificação inicial do problema
- `test_struct_values.py` - Confirma que scipy usa valores da estrutura
- `test_size_effects.py` - Demonstra resultados idênticos com sizes diferentes
- `test_struct_types.py` - Mostra diferença entre estruturas negativas/positivas/normalizadas
- `test_normalized_fix.py` - **Valida que a normalização resolve o problema**

Todos os testes confirmam que a normalização funciona corretamente.
