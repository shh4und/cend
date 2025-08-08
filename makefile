# Makefile para Avaliação de Reconstruções de Neurônios com DiademMetric

# --- Variáveis de Configuração ---
# Altere estes caminhos conforme a sua estrutura de diretórios.
# $(HOME) é uma maneira segura de se referir ao seu diretório home.
DIADEM_JAR    := ./metrics/DiademMetric/DiademMetric.jar
GOLD_DIR      := ./data/GoldStandardReconstructions
RESULTS_DIR   := ./results_swc
SCORES_FILE   := ./scores/scores.txt

# Número de imagens no dataset
N_IMAGES      := 9
INDICES       := $(shell seq 1 $(N_IMAGES))

# --- Alvos do Makefile ---

# O alvo padrão é 'all', que depende de 'evaluate'.
# Simplesmente digitar 'make' no terminal executará a avaliação.
.PHONY: all
all: evaluate

# Alvo para rodar a avaliação em todos os arquivos.
.PHONY: evaluate
evaluate:
	@echo "--- Iniciando a Avaliação das Reconstruções ---"
	@rm -f $(SCORES_FILE)
	
	@for i in $(INDICES); do \
		GOLD_SWC="$(GOLD_DIR)/OP_$${i}.swc"; \
		TEST_SWC="$(RESULTS_DIR)/OP_$${i}_reconstruction.swc"; \
		META_FILE="$(RESULTS_DIR)/OP_$${i}_reconstruction.meta"; \
		\
		echo "Avaliando: OP_$${i}"; \
		\
		if [ -f "$$TEST_SWC" ]; then \
			SCORE=$$(java -jar $(DIADEM_JAR) -G "$$GOLD_SWC" -T "$$TEST_SWC" -D 5 | grep 'Score:' | awk '{print $$2}'); \
			\
			if [ -f "$$META_FILE" ]; then \
				PARAMS=$$(cat "$$META_FILE" | tr '\n' ';' | sed 's/;/; /g' | sed 's/; $$//'); \
				echo "OP_$${i}; Score: $$SCORE; $$PARAMS" >> $(SCORES_FILE); \
			else \
				echo "OP_$${i}; Score: $$SCORE; Params: NOT_FOUND" >> $(SCORES_FILE); \
			fi; \
			echo "  -> Score = $$SCORE"; \
		else \
			echo "  -> ERRO: Arquivo não encontrado: $$TEST_SWC. Pulando."; \
			echo "OP_$${i}; Score: NOT_FOUND; Params: NOT_FOUND" >> $(SCORES_FILE); \
		fi \
	done
	@echo "--- Avaliação Concluída ---"
	@echo "Resultados salvos em: $(SCORES_FILE)"
	@echo "Conteúdo de $(SCORES_FILE):"
	@cat $(SCORES_FILE)

# Alvo para limpar os arquivos gerados.
# Rode 'make clean' para remover o arquivo de scores.
.PHONY: clean
clean:
	@echo "Limpando arquivo de scores gerado..."
	@rm -f $(SCORES_FILE)
	@echo "Limpeza concluída."