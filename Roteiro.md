  ---
  Roteiro — Tema 1: Regressão (20–25 min)

  ---
  1. A Dor do Negócio (~3 min)

  ▎ Abrir o app na primeira aba.

  "O mercado imobiliário de Bengaluru é um dos mais voláteis da Índia — dois apartamentos no mesmo andar podem ter preços completamente diferentes. A pergunta que queremos responder é: dado um imóvel com
  determinadas características, quanto ele vale? Isso é um problema de regressão supervisionada."

  Mostrar os KPIs do topo (13.320 registros, 9 colunas, faixa de preço).

  ---
  2. EDA (~5 min)

  ▎ Navegar para a aba EDA.

  - Distribuição de preços: "A distribuição é assimétrica à direita — a maioria dos imóveis custa até ₹150L, mas poucos imóveis de luxo puxam a média pra cima. Isso já nos alerta que o modelo vai errar mais
  nos extremos."
  - Área vs. Preço: "r = 0.54 — correlação moderada. Dois imóveis com a mesma metragem podem ter preços muito diferentes. Isso mostra que área sozinha não basta — precisamos de mais features."
  - Top 15 localizações: "Whitefield CBD cobra X vezes a média da cidade por sqft. Localização é o principal driver de preço — e vai ser nossa feature mais importante."
  - Heatmap: "total_sqft e bath são as variáveis numéricas mais correlacionadas com preço. bhk e bath se correlacionam muito entre si — multicolinearidade que o Ridge vai lidar bem."

  ---
  3. Pré-processamento (~4 min)

  ▎ Navegar para a aba Pré-processamento.

  Percorrer a lista de 8 etapas rapidamente:

  "Removemos a coluna society (41% nulos, zero poder preditivo). Extraímos BHK do texto, convertemos intervalos de sqft para média, removemos outliers por bairro. 1.287 bairros viraram 241 — raros agrupados em
   'other'. Por fim, One-Hot Encoding transformou localização em ~241 colunas binárias."

  - Mostrar o gráfico de BHK: "2 e 3 BHK dominam o mercado."
  - Mostrar Feature Selection do Lasso: "O Lasso zerou X% das features automaticamente — ficamos só com o que realmente importa."

  ---
  4. Modelos & Avaliação (~7 min)

  ▎ Navegar para a aba Modelos & Avaliação.

  "Treinamos três modelos dentro de um sklearn Pipeline com StandardScaler — isso garante que Ridge e Lasso penalizem coeficientes na mesma escala, sem vazamento de dados no cross-validation."

  - Mostrar tabela de métricas: comparar R², MAE, RMSE e CV R² dos três modelos.
  - "O melhor modelo foi o [nome] com R² de X — explica X% da variância do preço."
  - Mostrar gráfico Previsto vs. Real: "Pontos próximos à diagonal = boas previsões."
  - Mostrar p-values: "Features com p < 0.05 são estatisticamente significativas. Localização domina os coeficientes mais altos."

  ---
  5. Simulador (~3 min)

  ▎ Navegar para a aba Simulador.

  "Agora a parte prática — vamos simular um imóvel real."

  Configurar ao vivo: bairro popular (ex: Whitefield), 1200 sqft, 2 BHK, 2 banheiros → clicar Calcular.

  "O modelo retorna ₹X Lakhs. Comparando com a mediana do bairro, esse imóvel está [acima/abaixo] — o app já diz se é uma boa oportunidade ou não."

  ---
  Fechamento (~2 min)

  "Regressão linear com regularização Ridge/Lasso, processamento robusto de dados e 241 localizações em One-Hot Encoding nos deram um modelo com R² de X. O próximo passo natural seria testar Random Forest ou
  XGBoost para capturar relações não-lineares — mas isso já é Tema 2."