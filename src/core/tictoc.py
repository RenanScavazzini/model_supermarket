import time


def tictoc(tic, toc):
    """
    Descrição:
        Calcula o tempo decorrido entre dois instantes e retorna uma string formatada com o tempo decorrido e os horários de início e fim.
    
    Parâmetros:
        tic (float): Timestamp inicial (em segundos desde a época Unix).
        toc (float): Timestamp final (em segundos desde a época Unix).
    
    Retorno:
        str: String formatada com o tempo decorrido e os horários de início e fim.
    
    Referências:
        ---
    
    Autor:
        Renan Douglas Floriano Scavazzini
        Email: renanscavazzini@gmail.com

    Versão:
        1.0 - 05/05/2026

    Copyright:
        Copyright (c) 2026 Renan Douglas Floriano Scavazzini
    """
    p1 = time.strftime("%H:%M:%S", time.gmtime(toc - tic))
    p2 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(tic - 10800))
    p3 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(toc - 10800))
    return f'({p1}) [{p2} - [{p3}]'