# ==============================================================================
# Makefile para Avaliação de Reconstruções de Neurônios com DiademMetric
#
# Este Makefile automatiza a comparação entre os SWC gerados pelo pipeline
# e um conjunto de referência (Gold Standard), produzindo um arquivo CSV
# com os scores e os parâmetros utilizados em cada reconstrução.
# ==============================================================================

# --- Variáveis de Configuração ---
# Altere estes caminhos conforme a sua estrutura de diretórios.
DIADEM_JAR    := ./metrics/DiademMetric/DiademMetric.jar
GOLD_DIR      := ./data/GoldStandardReconstructions
RESULTS_DIR   := ./results_swc
SCORES_DIR    := ./scores
SCORES_FILE   := $(SCORES_DIR)/scores.csv

# Parâmetros do Dataset
N_IMAGES      := 9
INDICES       := $(shell seq 1 $(N_IMAGES))

# Cabeçalho para o arquivo CSV de saída. A ordem aqui deve corresponder à ordem de extração abaixo.
CSV_HEADER    := "ImageID,DiademScore,SourceImage,SigMax,SigMin,SigStep,NeuronThreshold,PruningThreshold,SmoothingFactor,NumPointsPerBranch"

# --- Alvos Principais ---

# O alvo padrão é 'all', que executa a avaliação.
# Digitar 'make' no terminal iniciará o processo.
.PHONY: all
all: evaluate

# Alvo para rodar a avaliação em todos os arquivos e gerar o CSV.
.PHONY: evaluate
evaluate:
	@echo "--- Iniciando a Avaliação das Reconstruções ---"
	# Garante que o diretório de scores exista
	@mkdir -p $(SCORES_DIR)
	# Cria o arquivo CSV e insere o cabeçalho
	@echo $(CSV_HEADER) > $(SCORES_FILE)
	
	@echo "Avaliando $(N_IMAGES) imagens..."
	@for i in $(INDICES); do \
		# Define os caminhos dos arquivos para a iteração atual
		GOLD_SWC="$(GOLD_DIR)/OP_$${i}.swc"; \
		TEST_SWC="$(RESULTS_DIR)/OP_$${i}_reconstruction.swc"; \
		META_FILE="$(RESULTS_DIR)/OP_$${i}_reconstruction.meta"; \
		IMAGE_ID="OP_$${i}"; \
		\
		# Verifica se o arquivo de reconstrução existe
		if [ -f "$$TEST_SWC" ]; then \
			# Roda o DiademMetric e extrai o score.
			# O '|| echo "ERROR"' garante que o comando não falhe se o grep não encontrar nada.
			SCORE=$$(java -jar $(DIADEM_JAR) -G "$$GOLD_SWC" -T "$$TEST_SWC" -D 5 | grep 'Score:' | awk '{print $$2}' || echo "ERROR"); \
			\
			# Se o arquivo de metadados existir, extrai cada parâmetro.
			if [ -f "$$META_FILE" ]; then \
				# Usamos grep e awk para extrair o valor de cada parâmetro de forma robusta.
				SOURCE_IMG=$$(grep 'source_image:' "$$META_FILE" | awk '{print $$2}'); \
				SIG_MAX=$$(grep 'sig_max:' "$$META_FILE" | awk '{print $$2}'); \
				SIG_MIN=$$(grep 'sig_min:' "$$META_FILE" | awk '{print $$2}'); \
				SIG_STEP=$$(grep 'sig_step:' "$$META_FILE" | awk '{print $$2}'); \
				NEURON_THRESH=$$(grep 'neuron_threshold:' "$$META_FILE" | awk '{print $$2}'); \
				PRUNING_THRESH=$$(grep 'pruning_threshold:' "$$META_FILE" | awk '{print $$2}'); \
				SMOOTH_FACTOR=$$(grep 'smoothing_factor:' "$$META_FILE" | awk '{print $$2}'); \
				NUM_POINTS=$$(grep 'num_points_per_branch:' "$$META_FILE" | awk '{print $$2}'); \
				\
				# Monta a linha do CSV com os valores extraídos
				CSV_LINE="$$IMAGE_ID,$$SCORE,$$SOURCE_IMG,$$SIG_MAX,$$SIG_MIN,$$SIG_STEP,$$NEURON_THRESH,$$PRUNING_THRESH,$$SMOOTH_FACTOR,$$NUM_POINTS"; \
			else \
				# Caso o arquivo .meta não seja encontrado
				CSV_LINE="$$IMAGE_ID,$$SCORE,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND"; \
			fi; \
			echo "  -> $$IMAGE_ID | Score: $$SCORE"; \
		else \
			# Caso o arquivo .swc de reconstrução não seja encontrado
			echo "  -> ERRO: Arquivo não encontrado: $$TEST_SWC. Pulando."; \
			CSV_LINE="$$IMAGE_ID,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND,NOT_FOUND"; \
		fi; \
		# Adiciona a linha construída ao arquivo CSV
		echo "$$CSV_LINE" >> $(SCORES_FILE); \
	done
	@echo "--- Avaliação Concluída ---"
	@echo "Resultados salvos em: $(SCORES_FILE)"
	@echo "--- Conteúdo de $(SCORES_FILE) ---"
	@cat $(SCORES_FILE) | column -s, -t

# Alvo para limpar os arquivos gerados.
# Rode 'make clean' para remover o diretório e o arquivo de scores.
.PHONY: clean
clean:
	@echo "Limpando arquivo de scores gerado..."
	@rm -f $(SCORES_FILE)
	@echo "Limpeza concluída."