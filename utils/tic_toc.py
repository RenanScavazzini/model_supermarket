import time


def tictoc(tic, toc):
    p1 = time.strftime("%H:%M:%S", time.gmtime(toc - tic))
    p2 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(tic - 10800))
    p3 = time.strftime("%d/%m/%Y %H:%M:%S", time.gmtime(toc - 10800))
    return f'({p1}) [{p2} - [{p3}]'