from numpy import log as ln
ln(0.05)/(-0.75)
b = list(range(10))
b

a=["31","122","333","44","335"]
c = list(range(1, len(a)+1))
c[-5]
range(len(a))

for i in range(0, len(a)):  # 4주 or 5주에 따라 클래스가 생성되야함.
    if i == 0:
        print("a=0",i)
    elif i == 1:
        print("a=1",i)
    elif i == 2:
        print("a=2",i)
    elif i == 3:
        print("a=3", i)
    else:
        print("a=4", i)

import multiprocessing
import threading
def work(x):
    print(x)
if __name__ == "__main__":
    p0_thread = threading.Thread(target=)
    p0_thread = multith(target=work(5))
    p0_thread.start()
