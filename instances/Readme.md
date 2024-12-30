# Formato

Os arquivos tem o seguinte formato. Os números de linha abaixo se referem a linhas não vazias.

- Linha 1: Número de GPUs (n).
- Linha 2: Quantidade de VRAM V (que é a mesma para todos os GPUs).
- Linha 3: Número de tipos diferentes (|T|).
- Linha 4: Número de PRNs m.
- Linhas 4+1,...,4+m descrevem cada uma das m PRNs. Cada linha contêm, nesta ordem, os seguintes valores (na forma de números inteiros positivos separados por espaços): tipo da PRN t_j (valor de 1 até |T|) e o consumo de VRAM v_j (valor de 1 até V, mas geralmente muito menor que V).
