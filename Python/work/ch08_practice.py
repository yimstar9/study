import os
print('\n 현재 경로:',os.getcwd())

try:
    ftest1=open(r'data\sample.txt',mode='r')
    while True:
        m=ftest1.readline()
        if not m: break
        print(m.strip())

    fopen2=open(r'data\ftest2.txt',mode='w')
    fopen2.write('my first text is')
    fopen3=open(r'data\ftest2.txt',mode='a')
    fopen3.write('\nmy second txt')
except Exception as e:
    print('에러발생:',e)

finally:
    ftest1.close()
    fopen2.close()
    fopen3.close()

####
import os

try:
    ftest=open(r'data\ftest.txt',mode='r')  #read로 전체 읽기
    full_txt = ftest.read()
    print(full_txt)
    print(type(full_txt))

    ftest1=open(r'data\ftest.txt',mode='r') #readlines으로 읽기
    line = ftest1.readlines()
    print(line)
    lines=[]
    for l in line:
        lines.append(l.strip())
    print(lines)

    ftest2=open(r'data\ftest.txt',mode='r')
    line1 = ftest2.readline()
    print(line1)
except Exception as e:
    print('Error 발생 :',e)
finally:
    ftest.close()
    ftest1.close()


###############
import os

print(os.getcwd())
txt_data = 'data/'
sub_dir=os.listdir(txt_data)
print(sub_dir)

#각 디렉터리 텍스트 자료 수집 함수
def txtpro(sub_dir):
    first_txt=[]
    second_txt=[]
    file_list = ""
    #디렉터리 구성
    for sdir in sub_dir:
        if (os.path.isdir(txt_data + sdir)):
            dirname = txt_data +sdir
            file_list = os.listdir(dirname)
        #파일구성
        for fname in file_list:
            file_path=dirname+'\\'+fname
            if os.path.isfile(file_path):
                try:
                    file=open(file_path,'r')
                    if sdir =='first':
                        first_txt.append((file.read()))
                    else:
                        second_txt.append(file.read())
                except Exception as e:
                    print('예외발생 :',e)
                finally:
                    file.close()
    return first_txt, second_txt

first_txt, second_txt= txtpro(sub_dir)
print('first_txt 길이 =',len(first_txt))
print('second_txt 길이 =',len(second_txt))

tot_txt = first_txt+second_txt
print('tot_txt길이 :',len(tot_txt))

print(tot_txt)
print(type(tot_txt))

#######################
import pandas as pd
import os
from statistics import mean


print(os.getcwd())

score = pd.read_csv('data\data.csv')

print(score.info())
print(score.head())

dur = score.Duration
pul = score['Pulse']
max = score['Maxpulse']
cal = score['Calories']
print(dur.max())
print(pul.max())
print(mean(dur))
print(min(dur))
print(min(cal))
print(dur.min())
dept_count= {}

for key in pul:
    dept_count[key]=dept_count.get(key,0)+1

print(dept_count)


###
import pickle
pfile_r = open('ch8_data/ch8_data/data/tot_texts.pck',mode='rb')
# pickle.dump(tot_texts,pfile_w)
tot_text_read=pickle.load(pfile_r)
print(len(tot_text_read))
print(type(tot_text_read))
print(tot_text_read)

### 이미지 파일 이동
import os
from glob import glob
print(os.getcwd())
img_path='ch8_data/ch8_data/images/'
img_path2='ch8_data/ch8_data/images2/'

if os.path.exists(img_path):
    print('해당 디렉터리가 존재함')
    img=[]
    os.mkdir(img_path2)

    for pic_path in glob(img_path+'*.png'):
        img_path=os.path.split(pic_path)
        img.append(img_path[1])
        #이미지 파일 읽기
        rfile=open(file=pic_path,mode='rb')
        ouput = rfile.read()
        #이미지 파일 이동(새로운 폴더에 쓰기)
        wfile = open(img_path2+img_path[1],mode='wb')
        wfile.write(ouput)
    rfile.close()
    wfile.close()
else:
    print('해당 디렉터리 없음')
print(img)