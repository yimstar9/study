for i in range(1,6):
    for j in range(i):
        print("*",end='')
    print("")
##########################
for i in range(1,6):
    for j in range(5-i):
        print(" ",end='')
    for k in range(i):
        print("*",end='')
    print("")
##########################
for i in range(1,6):
    for j in range(6,i,-1):
        print("*",end='')
    print("")
##########################
for i in range(1,6):
    for j in range(5,i,-1):
        print(" ",end='')
    for k in range(i*2-1):
        print("*",end='')
    print("")
##############################
for i in range(1,6):
    print(f'{"*"*(i):>5}')
##############################
for i in range(5,0,-1):
    print(f'{"*" * (i)}')
##############################
for i in range(1,6):
    print(f'{"*"*(2*i-1):^9}')
##############################
i=0
while True:
    i+=1
    if(i>5):break
    print(f'{"*"*(i):>5}')
##############################
i=0
while True:
    i+=1
    if(i>5):break
    print(f'{"*"*(i)}')
##############################
i=0
while True:
    i+=1
    if(i>5):break
    print(f'{"*" * (6-i)}')
#############################
i = 0
while True:
    i += 1
    if (i > 5): break
    print(f'{"*" * (2 * i - 1):^9}')
##############################