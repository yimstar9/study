#!/usr/bin/env python
# coding: utf-8

# # 02 최소한의 도구로 딥러닝을 시작합니다

# 이 노트북을 주피터 노트북 뷰어(nbviewer.jupyter.org)로 보거나 구글 코랩(colab.research.google.com)에서 실행할 수 있습니다.
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://nbviewer.org/github/rickiepark/do-it-dl/blob/master/Ch02.ipynb"><img src="https://jupyter.org/assets/share.png" width="60" />주피터 노트북 뷰어로 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/rickiepark/do-it-dl/blob/master/Ch02.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
# </table>

# ## 02-2 딥러닝을 위한 도구들을 알아봅니다

# In[1]:


my_list = [10, 'hello list', 20]
print(my_list[1])

# In[2]:


my_list_2 = [[10, 20, 30], [40, 50, 60]]
print(my_list_2[1][1])

# In[3]:


import numpy as np
print(np.__version__)

# In[4]:


my_arr = np.array([[10, 20, 30], [40, 50, 60]])
print(my_arr)

# In[5]:


type(my_arr)

# In[6]:


my_arr[0][2]

# In[7]:


np.sum(my_arr)

# <퀴즈>
# `my_arr` 배열의 두 번째 행의 첫 번째 원소를 `print()` 함수로 출력해 보세요.

# In[8]:


print(my_arr[1][0])

# In[9]:


import matplotlib.pyplot as plt

# In[10]:


plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) # x 좌표와 y 좌표를 파이썬 리스트로 전달합니다.
plt.show()

# In[11]:


plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])
plt.show()

# In[12]:


x = np.random.randn(1000) # 표준 정규 분포를 따르는 난수 1,000개를 만듭니다.
y = np.random.randn(1000) # 표준 정규 분포를 따르는 난수 1,000개를 만듭니다.
plt.scatter(x, y)
plt.show()
