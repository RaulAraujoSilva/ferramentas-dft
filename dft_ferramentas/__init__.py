"""dft_ferramentas — ferramentas abertas para o Dimensionamento da Força de Trabalho (DFT) no setor público.

Módulos (um por etapa do método; cada um com docstring, referências e exemplo em exemplos/):
    capacidade      1. mapeamento de processos e análise de capacidade (lei de Little, utilização, gargalo)
    recursos        2. identificação de recursos por atividade (matriz atividade × recurso, horas por perfil)
    distribuicoes   2.1 curva de distribuição de tempo das atividades (ajuste, testes de aderência, PERT/3 pontos)
    nivelamento     3. nivelamento de recursos e teoria das restrições (5 passos, tambor-pulmão-corda)
    previsao        4. previsão de demanda (métodos de referência, SES/Holt/Holt amortecido/Holt-Winters, erros)
    simulacao       5. simulação de processos por Monte Carlo (esforço por entrega, P80, fila/processo)
Exemplos SEMPRE com dados fictícios. Números canônicos do curso conferidos em tests/.
"""
__version__ = "0.1.0"
