# Prova Técnica — Visão Computacional (TCS Industrial)

Processo seletivo 2025 — área de Visão Computacional.


## Sobre o uso de IA

Utilizei IA como apoio para estudar os conceitos e a sintaxe das bibliotecas antes de escrever cada etapa. As decisões técnicas, os testes e os diagnósticos dos problemas descritos abaixo foram feitos por mim, com base nos resultados que observei rodando o código.


----

## Questão 1 — Detecção e contagem com YOLO

*preencher*
---

## Questão 2 — Segmentação com visão tradicional

### Abordagem

Utilizei a imagem de moedas do próprio scikit-image (`skimage.data.coins()`). O pipeline segue quatro etapas: limiarização com o método de Otsu, limpeza com operações morfológicas (abertura e fechamento), separação de moedas encostadas com watershed, e um filtro final por área para remover um artefato que sobreviveu às etapas anteriores.

### Resultado

*23 moedas contadas* na segmentação final, enquanto a imagem tem 24 moedas reais. Todas as limitações para alcançar as 24 moedas observadas por mim estão explicadas abaixo.

![Segmentação final](saida_q2_segmentada.png)

### Etapas

**1. Inspeção inicial**
Antes de processar, inspecionei a imagem para confirmar sua estrutura: é uma matriz numpy de 303×384 (linhas × colunas), com valores de 0 a 255 (padrão `uint8`), variando entre 1 e 252 nessa imagem específica. Também olhei a imagem original para entender a distribuição de iluminação antes de segmentar.


**2. Limiarização com Otsu**
Apliquei o método de Otsu, que testa todos os limiares possíveis e escolhe o que melhor separa os pixels em dois grupos (fundo e objeto), com base no histograma da própria imagem. O limiar encontrado foi *107*. Como o método decide olhando só o brilho de cada pixel, isoladamente, uma região do fundo no canto superior esquerdo, mais clara que o restante do fundo, por iluminação não-uniforme, ficou acima do limiar e foi classificada como objeto, virando uma mancha branca na imagem binarizada.

![Histograma da imagem](histograma.png)
![Imagem binarizada](binarizada.png)


**3. Limpeza morfológica**
Apliquei fechamento e depois abertura, usando `disk(3)` como elemento estruturante. Essas operações percorrem a imagem comparando cada pixel central com sua vizinhança (definida pelo formato do elemento estruturante) e decidem se ele deve mudar de valor. O objetivo era fechar buracos pequenos dentro das moedas e remover ruído, mas a mancha era grande demais (bem maior que o elemento estruturante) o que significa que se eu tentasse modificar o raio eu, querendo ou não, afetaria outras moedas também.

![Depois da limpeza morfológica](limpa.png)


**4. Separação com watershed**
Como a morfologia sozinha não separa objetos encostados nem elimina a mancha, apliquei watershed. O método calcula, para cada pixel de objeto, a distância até a borda mais próxima (transformada de distância), o centro de cada moeda vira o ponto mais "alto" desse relevo, como se fosse um pico. A partir de marcadores nesses centros, o algoritmo "inunda" para fora até que as águas de duas moedas vizinhas se encontrem, e é aí que a fronteira entre elas é desenhada. Para gerar os marcadores, foi necessário reorganizar as coordenadas dos picos encontrados no formato que o numpy espera (listas separadas de índices de linha e de coluna).

Resultado nessa etapa: **25 regiões contadas** — 24 moedas reais + a mancha, que também gerou um pico e foi tratada como um objeto (por isso teve um objeto a mais).

![Segmentação por watershed](watershed.png)


**5. Filtro por área**
Para remover a mancha, medi a área (em pixels) de cada uma das 25 regiões encontradas. A maioria das moedas ficou entre aproximadamente 1100 e 3100 pixels. Duas regiões destoavam claramente: uma com 12 pixels (ruído pontual) e outra com 9668 pixels (a mancha, mais de 6 vezes o tamanho de uma moeda). Descartei as regiões fora de uma faixa de 500 a 5000 pixels, chegando a **23 moedas**. (Imagem que foi enviada para vocês).


### Justificativas

**Por que esse método de limiarização?**
Escolhi o Otsu porque ele calcula o limiar automaticamente a partir do histograma de cada imagem, em vez de depender de um valor fixo escolhido manualmente. Ele testa todos os limiares possíveis e escolhe o que melhor separa os pixels em dois grupos, com base em quão distintos esses grupos ficam entre si.

**O que aconteceria se a iluminação da cena mudasse?**
Testei isso escurecendo a imagem artificialmente (multiplicando os pixels por 0,5). O limiar do Otsu caiu de 107 para 53 — quase exatamente pela metade, o que confirma que o método se readapta ao novo histograma. Porém, o contraste entre moeda e fundo diminuiu, e pequenas variações internas das moedas passaram a cruzar o novo limiar, criando buracos que não existiam antes. A mancha do canto, por ser originalmente muito clara, continuou sendo classificada como objeto mesmo na versão escurecida.

**Cite um caso em que seu método falharia**
Aqui posso citar justamente a falha que encontrei ao longo do desenvolvimente dessa questão, que foi uma região do fundo mais clara que o restante (por reflexo ou iluminação não-uniforme) é classificada como objeto, porque a limiarização decide olhando só o brilho de cada pixel, sem nenhuma noção de posição ou contexto. Esse tipo de erro não é corrigido pela morfologia quando a região afetada é grande, e pode até "contaminar" outras etapas do pipeline (ver abaixo).

### Limitações observadas

O pipeline não chegou a 24/24. Ao investigar, percebi que uma das moedas reais provavelmente ficou fundida com a mancha durante a etapa de fechamento (a dilatação aproximou as bordas das duas regiões o suficiente para uni-las). Como a região resultante ficou grande demais, ela foi descartada inteira pelo filtro de área. Com mais tempo, tentaria resolver isso reduzindo o raio do fechamento ou filtrando a mancha por formato (bordas irregulares) em vez de só por área.

---



## Questão 3 — Tanque com espuma

*A preencher.*
