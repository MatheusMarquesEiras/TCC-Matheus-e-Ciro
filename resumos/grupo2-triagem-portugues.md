# Trabalhos relacionados — Grupo 2
## Triagem médica e modelos em português

Este grupo reúne cinco trabalhos mais próximos do problema central do TCC: usar linguagem natural para triar pacientes e adaptar modelos para o português. Dois deles trabalham diretamente com português, e um resolve exatamente a mesma tarefa (queixa livre para departamento) com outra técnica.

---

## 1. Stewart et al. (2023)

**Título completo.** Applications of Natural Language Processing at Emergency Department Triage: A Narrative Review
**Publicação.** PLoS ONE, v. 18, n. 12, e0279953

### Contexto e problema
A triagem de pronto-socorro precisa priorizar pacientes rápido e com pouca informação. Os autores quiseram mapear tudo que já foi feito usando PLN sobre os dados coletados na triagem e responder três perguntas. Como o PLN foi aplicado a esses dados, se os modelos de PLN superam humanos ou técnicas atuais de estratificação de risco, e se usar texto livre melhora a previsão em comparação com usar só dados estruturados.

### Método
É uma revisão de escopo (scoping review). Incluíram toda pesquisa revisada por pares, em inglês, que aplicou alguma técnica de PLN a texto livre obtido na triagem. Excluíram estudos só de vigilância de doença e estudos que usaram informação obtida depois da triagem. Buscaram nas bases MEDLINE, Embase, Cochrane, Web of Science e Scopus, com última busca em janeiro de 2022.

### Resultados
Modelos que usam texto livre tendem a prever desfechos melhor do que modelos que usam apenas dados estruturados. Apontaram como principais barreiras a falta de datasets públicos e a ausência de validação prospectiva em ambiente real.

### Limitações
É uma revisão, não um método próprio. Restringe-se ao contexto de pronto-socorro e a publicações em inglês.

### Relevância para o TCC
Dá o panorama geral do campo e sustenta a decisão de usar a queixa em texto livre como entrada do modelo, em vez de formulários estruturados.

---

## 2. Wang et al. (2023)

**Título completo.** A Disease-Prediction Protocol Integrating Triage Priority and BERT-Based Transfer Learning for Intelligent Triage
**Publicação.** Bioengineering, v. 10, n. 4, artigo 420

### Contexto e problema
Em hospitais grandes, com muitas especialidades e subespecialidades, o paciente muitas vezes não sabe a qual departamento ir e acaba na fila errada. Os autores queriam um sistema de triagem que o próprio paciente pudesse usar, prevendo o diagnóstico e o departamento certo a partir da queixa.

### Dados e modelo
Trabalharam com textos médicos neurológicos em chinês, um problema multirrótulo (um registro pode ter várias doenças e departamentos). Usaram o BERT com transfer learning para classificar o texto da queixa principal.

### Método
A contribuição central é o método Triage Priority (TP), que rotula as combinações de diagnóstico encontradas nos registros e converte o problema multirrótulo em um problema de rótulo único, levando em conta a gravidade da doença e reduzindo a sobreposição de classes do dataset. Para tratar o desbalanceamento, adicionaram uma função de perda composta baseada em aprendizado sensível a custo sobre a arquitetura do BERT.

### Resultados
O método TP atingiu 87,47% de acurácia na classificação dos textos, superando outras formas de transformar o problema multirrótulo. Com a função de perda composta, a acurácia subiu para 88,38%, superando outras funções de perda. Tudo isso sem aumentar muito a complexidade em relação aos métodos tradicionais.

### Limitações
Depende do método TP para reduzir o problema a rótulo único, o que pode perder informação em casos genuinamente multirrótulo. É específico para textos neurológicos em chinês.

### Relevância para o TCC
É o trabalho mais próximo do problema deste TCC: classificar especialidade ou departamento a partir de queixa em texto livre. As diferenças principais são o modelo (BERT contra LLM com QLoRA), o idioma (chinês contra português) e o tratamento de desbalanceamento, um problema que também aparece no MedPT.

---

## 3. SmartTriage — Valmianski et al. (2021)

**Título completo.** SmartTriage: A System for Personalized Patient Data Capture, Documentation Generation, and Decision Support
**Publicação.** Proceedings of Machine Learning for Health (ML4H) 2021
**Instituição.** Kaiser Permanente

### Contexto e problema
Verificadores de sintomas online viraram ferramenta importante, ainda mais depois que a pandemia empurrou até 90% das consultas de atenção primária para o atendimento virtual. Os autores queriam ir além do verificador comum, integrando o sistema de triagem ao prontuário eletrônico.

### Dados
Usaram conjuntos de dados enormes: mais de 25 milhões de encontros de atenção primária e 1 milhão de entradas de texto livre com o motivo da visita.

### Método
O SmartTriage identifica a queixa principal a partir do texto livre e faz uma sequência de perguntas objetivas para coletar os sintomas relevantes, condicionadas ao histórico do paciente vindo do prontuário. É composto por quatro modelos: uma LSTM para representar o histórico do paciente, um transformer ajustado para extrair a queixa principal, uma random forest para decidir a sequência de perguntas, e uma rede feed-forward para as previsões de suporte à decisão clínica.

### Resultados
O sistema prevê códigos ICD-10-CM detalhados e também pedidos de medicação, exames de laboratório e imagem. As respostas do paciente e as previsões de suporte à decisão voltam para o prontuário. No total, cobre 337 queixas principais, que juntas representam mais de 90% de todos os encontros de atenção primária.

### Limitações
É um sistema industrial complexo, com quatro componentes e integração pesada com prontuário, difícil de reproduzir fora daquele contexto. Depende de volume de dados que poucas instituições têm.

### Relevância para o TCC
Mostra um sistema de triagem completo e de ponta a ponta rodando em produção. Serve de contraste: este TCC foca só no componente de classificação de especialidade, com um único modelo compacto, em vez de uma arquitetura de quatro modelos.

---

## 4. Paiola et al. (2024)

**Título completo.** Adapting LLMs for the Medical Domain in Portuguese: A Study on Fine-Tuning and Model Evaluation
**Instituição.** Universidade Estadual Paulista (UNESP), Bauru

### Contexto e problema
LLMs médicos eram quase todos em inglês. Os autores queriam avaliar LLMs como agentes médicos em português e caminhar para um assistente virtual confiável para profissionais de saúde no Brasil.

### Dados e modelo
Usaram os datasets HealthCareMagic-100k-en e MedQuAD, traduzidos do inglês para o português com o GPT-3.5. Fizeram fine-tuning do modelo ChatBode-7B, um modelo de 7 bilhões de parâmetros focado em português.

### Método
Aplicaram o método PEFT-QLoRA para adaptar o ChatBode-7B, reduzindo o custo do fine-tuning. Compararam também com o InternLM2, que já tinha treino inicial em dados médicos. Avaliaram os modelos em métricas como acurácia, completude, segurança, gramaticalidade e coerência.

### Resultados
O InternLM2, com treino inicial em dados médicos, teve o melhor desempenho geral, com boa precisão e adequação. Os modelos DrBode, derivados do ChatBode, sofreram catastrophic forgetting do conhecimento médico adquirido, ou seja, esqueceram parte do que sabiam ao serem ajustados. Mesmo assim, foram melhores em gramaticalidade e coerência. Um desafio grande foi a baixa concordância entre avaliadores, mostrando a necessidade de protocolos de avaliação mais robustos.

### Limitações
Dados traduzidos por máquina podem não capturar nuances clínicas e culturais do português. A avaliação humana teve baixa concordância entre juízes, o que enfraquece as conclusões.

### Relevância para o TCC
É o trabalho mais alinhado ao contexto deste TCC: mesma técnica (QLoRA), mesmo idioma (português), mesmo objetivo geral (modelo médico). A diferença é a tarefa, um assistente conversacional lá contra classificação de especialidade aqui. O problema deles com dados traduzidos reforça a vantagem de usar o MedPT, que é feito de dados reais em português brasileiro. O problema deles com concordância entre juízes motivou o cuidado na avaliação qualitativa deste TCC.

---

## 5. Elvas, Santos e Ferreira (2025)

**Título completo.** Fine-tuning of Language Models for Automated Structuring of Medical Exam Reports to Improve Patient Screening and Analysis
**Publicação.** Scientific Reports (Nature), artigo 41598-025-05695-6

### Contexto e problema
Laudos de exames de imagem costumam vir em texto livre não estruturado, e organizá-los para triagem é trabalhoso. Os autores queriam automatizar a análise de laudos de cardiologia em português europeu para agilizar a triagem de pacientes.

### Dados e modelo
Trabalharam com laudos de cardiologia em português europeu. Ajustaram o modelo MediAlbertina PT-PT, uma variante do BERT voltada para o português.

### Método
Montaram o dataset com tokenização, etiquetagem gramatical (part-of-speech) e anotação manual. Fizeram fine-tuning do MediAlbertina na tarefa de reconhecimento de entidades nomeadas (NER), que identifica no texto termos como condições e achados clínicos.

### Resultados
Atingiram 96,13% de acurácia no reconhecimento de entidades. O sistema permite identificar rapidamente condições como estenose aórtica por uma interface interativa, reduzindo bastante o tempo de revisão manual e ajudando no monitoramento de pacientes e na alocação de recursos.

### Limitações
É português europeu, não brasileiro, e restrito a laudos de cardiologia. NER é uma tarefa de extração, diferente da classificação de especialidade deste TCC.

### Relevância para o TCC
Prova de que fine-tuning de modelos de linguagem em português entrega alta acurácia em tarefas de texto clínico. O resultado deles (96%) e o deste TCC (85%) são de tarefas diferentes, mas mostram a mesma tendência: modelos ajustados funcionam bem em saúde em português.

---

## Síntese do grupo

| Trabalho | Idioma | Modelo | Tarefa | Resultado |
|----------|--------|--------|--------|-----------|
| Stewart (2023) | Inglês | Revisão | Panorama de PLN na triagem | Texto livre supera dados estruturados |
| Wang (2023) | Chinês | BERT + TP | Queixa para departamento | 88,38% acurácia |
| SmartTriage (2021) | Inglês | 4 modelos | Triagem completa | 337 queixas, >90% dos casos |
| Paiola (2024) | Português | ChatBode-7B (QLoRA) | Assistente médico | InternLM2 melhor, forgetting no DrBode |
| Elvas (2025) | Português (PT) | MediAlbertina | NER em laudos | 96,13% acurácia |

Os dois trabalhos mais próximos são Wang et al. (mesma tarefa, técnica diferente) e Paiola et al. (mesma técnica e idioma, tarefa diferente). Este TCC fica exatamente no cruzamento dos dois: mesma tarefa de classificar especialidade a partir de queixa livre, com a mesma técnica de QLoRA, no português brasileiro, usando um dataset real (MedPT).
