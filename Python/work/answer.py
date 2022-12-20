import os

def change_dir(path):
    os.chdir(path)

print('경로 변경 전 %s' % os.getcwd())
change_dir('E:\GoogleDrive\workplace\Python\work')
print('경로 변경 후 %s' % os.getcwd())