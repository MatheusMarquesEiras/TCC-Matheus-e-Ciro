# Trabalhos relacionados — Grupo 1
## Diálogo médico e documentação clínica

Este grupo reúne quatro trabalhos que usam LLMs para conversar com o paciente, resumir a consulta e gerar documentação clínica. Todos servem de base para justificar por que fine-tuning de modelos de linguagem funciona bem no domínio médico.

---

## 1. ChatDoctor — Li et al. (2023)

**Título completo.** ChatDoctor: A Medical Chat Model Fine-Tuned on a Large Language Model Meta-AI (LLaMA) Using Medical Domain Knowledge
**Publicação.** Cureus, v. 15, n. 6, e40895. DOI 10.7759/cureus.40895

### Contexto e problema
Modelos genéricos como o ChatGPT tinham conhecimento médico raso e davam respostas clínicas incorretas com frequência. Os autores queriam um modelo especializado com maior precisão em conselhos médicos, num domínio onde erro tem custo alto.

### Dados
Usaram 100 mil diálogos reais entre pacientes e médicos coletados de uma plataforma online de consulta médica (HealthCareMagic). As conversas foram limpas e anonimizadas para preservar privacidade. Esse conjunto virou o dataset público HealthCareMagic-100k.

### Método
Fizeram fine-tuning do LLaMA nos 100 mil diálogos reais. O diferencial foi um mecanismo de recuperação de informação autodirigida, que deixa o modelo consultar fontes externas em tempo real, como a Wikipédia e bases médicas offline curadas, antes de responder. Assim o modelo combina o que aprendeu no treino com informação buscada na hora.

### Resultados
O fine-tuning com conversas reais melhorou bastante a capacidade do modelo de entender a necessidade do paciente e dar conselhos informados. A recuperação de informação externa aumentou ainda mais a precisão das respostas em comparação com o LLaMA base e o ChatGPT genérico. A avaliação foi feita por comparação de respostas, sem métrica automática única de referência.

### Limitações
A avaliação de qualidade médica é difícil de automatizar e depende de julgamento humano. O modelo é um assistente de conversa, não um classificador, então não reporta acurácia de rótulo.

### Relevância para o TCC
É a prova de conceito mais citada de que fine-tuning com diálogo médico real supera modelos genéricos. Inspirou a escolha de treinar em dados reais em português (MedPT) em vez de dados traduzidos ou sintéticos.

---

## 2. GatorTronGPT — Lyu et al. (2024)

**Título completo.** Automatic Summarization of Doctor-Patient Encounter Dialogues Using Large Language Model through Prompt Tuning
**Instituição.** University of Florida (arXiv 2403.13089)

### Contexto e problema
A carga de documentação clínica é uma das maiores causas de burnout médico. Resumir automaticamente a conversa entre médico e paciente ajuda, mas fazer fine-tuning completo de um LLM grande é caro. Os autores queriam um jeito barato de adaptar um LLM clínico grande para a tarefa de resumo.

### Dados e modelo
Usaram o GatorTronGPT, um LLM clínico generativo treinado com 277 bilhões de palavras (82 bilhões de texto clínico anonimizado do UF Health mais 195 bilhões de inglês geral), disponível em versões de até 20 bilhões de parâmetros. O benchmark de avaliação foi o MTS-DIALOG, um conjunto de diálogos clínicos.

### Método
Em vez de fine-tuning, usaram prompt tuning (soft prompts). A ideia é treinar apenas um pequeno vetor de prompt aprendível na entrada, mantendo todos os pesos do modelo congelados. Testaram estratégias de prompt tuning, tamanhos diferentes de soft prompt e capacidade de few-shot. Compararam com a solução anterior baseada em fine-tuning completo de um modelo T5-Large.

### Resultados
O GatorTronGPT-20B com prompt tuning teve o melhor desempenho em todas as métricas, superando tanto a versão de 5 bilhões quanto o T5-Large ajustado por fine-tuning completo. Números no resumo de diálogo médico-paciente foram ROUGE-1 de 0,3628, ROUGE-2 de 0,1549, ROUGE-L de 0,3472, BLEU de 0,3665 e BERTScore de 0,7309. O custo computacional é baixo porque os parâmetros do LLM não são atualizados durante o prompt tuning.

### Limitações
Prompt tuning depende de um modelo base já muito grande e bem treinado no domínio. As métricas de resumo (ROUGE, BLEU) medem sobreposição de texto, não correção clínica.

### Relevância para o TCC
Reforça o argumento central do trabalho: métodos de adaptação eficientes que congelam o modelo base (prompt tuning lá, QLoRA aqui) alcançam desempenho igual ou melhor que fine-tuning completo, com uma fração do custo. Justifica diretamente a escolha do QLoRA neste TCC.

---

## 3. Documentação clínica (MediGen) — Yi et al. (2024)

**Título completo.** Efficient Fine-Tuning of Large Language Models for Automated Medical Documentation
**Instituição.** University of Chicago (Data Science Institute)

### Contexto e problema
Para cada hora de cuidado direto ao paciente, o médico gasta quase duas horas em tarefas administrativas, principalmente preenchendo prontuário. Os autores queriam automatizar a geração de notas clínicas a partir da transcrição da consulta.

### Dados e modelo
Fizeram fine-tuning do LLaMA3-8B, um modelo aberto de 8 bilhões de parâmetros, gerando o modelo chamado MediGen. O treino usou pares de transcrição de consulta e a nota clínica correspondente no formato SOAP (Subjective, Objective, Assessment, Plan).

### Método
Aplicaram técnicas de fine-tuning eficiente sobre o LLaMA3-8B para transcrever e resumir a interação clínica em notas SOAP estruturadas. O foco foi manter alta acurácia minimizando o uso de recurso computacional, mesma preocupação deste TCC.

### Resultados
O MediGen atingiu ROUGE de 58% e BERTScore-F1 de 72% na geração de relatórios médicos, indicando notas precisas e clinicamente relevantes. Os autores concluem que a ferramenta tem potencial de reduzir bastante a carga administrativa do médico.

### Limitações
Gerar texto SOAP livre é uma tarefa aberta, mais difícil de avaliar do que classificação. As métricas usadas medem similaridade textual e não capturam totalmente a utilidade clínica.

### Relevância para o TCC
Mais um caso de fine-tuning eficiente de um LLaMA para tarefa médica estruturada, confirmando que modelos abertos de porte pequeno a médio dão conta de tarefas de texto clínico depois de ajustados.

---

## 4. MEDIQA-Chat 2023 — Ben Abacha et al. (2023)

**Título completo.** Overview of the MEDIQA-Chat 2023 Shared Tasks on the Summarization & Generation of Doctor-Patient Conversations
**Publicação.** Proceedings of the 5th Clinical NLP Workshop (ACL 2023), p. 503-513

### Contexto e problema
Gerar notas clínicas automaticamente a partir da conversa médico-paciente reduz a carga do médico e melhora a interação com o paciente. Para acelerar a pesquisa na área, os organizadores montaram uma competição científica aberta (shared task).

### Estrutura da competição
Foram três tarefas. A Tarefa A pede resumir um diálogo curto para gerar uma seção específica da nota clínica com seu cabeçalho. A Tarefa B trata do diálogo completo para nota completa. A Tarefa C inverte o problema e gera diálogos sintéticos a partir de notas clínicas, servindo para aumento de dados. Todos os datasets e códigos foram liberados publicamente.

### Participação e método
Dezessete equipes participaram e testaram uma ampla variedade de abordagens e modelos, de fine-tuning de modelos pré-treinados a modelos de sequência a sequência e prompting. A avaliação usou métricas automáticas que se correlacionam com julgamento humano.

### Resultados
As abordagens baseadas em fine-tuning de modelos pré-treinados dominaram o ranking. A geração de diálogos sintéticos (Tarefa C) se mostrou promissora para criar dados de treino em domínios com poucos exemplos reais.

### Limitações
É um artigo de visão geral de competição, não um método único. Os datasets são em inglês e focados em consulta primária.

### Relevância para o TCC
Mostra o estado da arte em compreensão de diálogo médico e reforça o valor de datasets reais de consulta. Ajuda a situar o MedPT como o equivalente em português brasileiro dessa linha de pesquisa.

---

## Síntese do grupo

| Trabalho | Modelo | Tarefa | Método de adaptação |
|----------|--------|--------|---------------------|
| ChatDoctor (2023) | LLaMA | Chat médico | Fine-tuning + recuperação externa |
| GatorTronGPT (2024) | GatorTronGPT 20B | Resumo de diálogo | Prompt tuning |
| MediGen (2024) | LLaMA3-8B | Notas SOAP | Fine-tuning eficiente |
| MEDIQA-Chat (2023) | Vários (17 equipes) | Resumo e geração | Diversos |

O fio condutor é claro. Fine-tuning e métodos eficientes de adaptação, sobre modelos abertos de porte pequeno a médio, entregam bom desempenho em tarefas médicas de texto. Este TCC segue a mesma linha, mas foca em classificação de especialidade em português com QLoRA.
