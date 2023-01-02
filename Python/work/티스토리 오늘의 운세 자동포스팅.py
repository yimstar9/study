
from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import re
from requests import get
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from work.config import tistory_config  #개인 정보(하드코딩) 모듈화



# tistory에 로그인을 합니다.
def tistory_login():
    # chrome_options = Options()
    # chrome_options.add_argument("--headless")  # chrome 띄워서 보려면 주석처리
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.implicitly_wait(3)
    driver.get('https://www.tistory.com/auth/login?redirectUrl=http%3A%2F%2Fwww.tistory.com%2F')
    driver.find_element(By.CSS_SELECTOR,"#cMain > div > div > div > a.btn_login.link_kakao_id").click()
    driver.find_element(By.NAME,'loginKey').send_keys(tistory_config.tistory_id)
    driver.find_element(By.NAME,'password').send_keys(tistory_config.tistory_pw)
    driver.find_element(By.NAME,'password').send_keys(Keys.ENTER)
    return driver

# authentication code 정보를 가져옵니다.
def get_authentication_code(driver, client_id, redirect_url):
    req_url = 'https://www.tistory.com/oauth/authorize?client_id=%s&redirect_uri=%s&response_type=code&state=someValue' % (client_id, redirect_url)
    driver.get(req_url)
    driver.find_element(By.XPATH,'//*[@id="contents"]/div[4]/button[1]').click()
    redirect_url = driver.current_url
    temp = re.split('code=', redirect_url)
    code = re.split('&state=', temp[1])[0]
    return code

# http://www.tistory.com/guide/api/index
# access token 정보를 가져옵니다.
def get_access_token(code, client_id, client_secret, redirect_url):
    url = 'https://www.tistory.com/oauth/access_token?'
    payload = {'client_id': client_id,
               'client_secret': client_secret,
               'redirect_uri': redirect_url,
               'code': code,
               'grant_type': 'authorization_code'}
    res = get(url, params=payload)
    token = res.text.split('=')[1]
    return token

#카테고리 이름 확인
def category_name(token):

    blogName = 'data-science'
    output = 'json'

    url = 'https://www.tistory.com/apis/category/list?'
    data = {
            'access_token': token,
            'output': output,
            'blogName': blogName,
            }
    r = requests.get(url, data)
    return r.text   #1093583

def page_url():
    # 웹페이지 크롤링
    html = urlopen("https://www.donga.com/news/List/Culture/unse?m=")
    bs = BeautifulSoup(html, "html.parser")
    unse=bs.select_one('#content > div:nth-child(3) > div.rightList > span.tit > a:nth-child(1)')
    unseurl=unse.get('href')
    unsetitle = unse.get_text()
    unsetitle = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", " ", unsetitle).strip()
    return unseurl,unsetitle

def image_url(unseurl):
    #이미지 크롤링
    html2 = urlopen(unseurl)
    bs2 = BeautifulSoup(html2, "html.parser")
    unseimg=bs2.select_one('#article_txt > div.articlePhotoC > span > img')
    unseimgurl=unseimg.get('src')
    return unseimgurl

def postWrite(title,imgurl):
    content = f'''<p><img src={imgurl}></p>'''
    url = 'https://www.tistory.com/apis/post/write?'
    data = {
             'access_token':token,
             'output': tistory_config.output,
             'blogName': tistory_config.blogName,
             'title': title,
             'content': content,
             'visibility': tistory_config.visibility,
             'category': tistory_config.category,
             'tag': tistory_config.tag,
             }
    r = requests.post(url, data=data)
    print ('자동 포스팅 성공')
    return r.text


if __name__ == '__main__':
    driver = tistory_login()
    code = get_authentication_code(driver, tistory_config.client_id, tistory_config.redirect_url)
    token = get_access_token(code, tistory_config.client_id, tistory_config.client_secret, tistory_config.redirect_url)
    print(token)
    unseurl,unsetitle  = page_url()
    unseimgurl = image_url(unseurl)
    postWrite(unsetitle,unseimgurl)