from multiprocessing import Pool
import time 
import datetime

def f(x):
    i = 0
    while True:
        print("Pool {} mit folgender Zeit {}".format(x,datetime.datetime.now()))
        time.sleep(2)
        i+=1
        if i == 10:
            break
    return True

if __name__ == '__main__':
    pool = Pool(5)
    print(pool.map(f,[1,2,3]))