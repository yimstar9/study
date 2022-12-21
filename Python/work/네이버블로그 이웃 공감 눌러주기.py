

import time
import pyperclip
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from webdriver_manager.chrome import ChromeDriverManager

id=input("id:")
pw=input("pw:")
id = id   #네이버 ID
pw = pw #네이버 패스워드
site_url = 'https://nid.naver.com/nidlogin.login?svctype=262144&amp;url=http://undefined/aside/'    #네이버 모바일 로그인 URL

#크롬 실행
def exec_chrom():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.maximize_window()
    return driver


#로그인
def login(driver):
    driver.get(site_url)
    pyperclip.copy(id)  # id를 클립보드에 복사
    driver.find_element(by='name', value='id').send_keys(Keys.CONTROL + 'v')  # id 입력창에 보냄
    pyperclip.copy(pw)  # 패스워드를 클립보드에 복사
    driver.find_element(by='name', value='pw').send_keys(Keys.CONTROL + 'v')  # 패스워드를 입력창에 보냄
    driver.find_element(by='xpath', value='//*[@id="log.login"]').click()  # 로그인 버튼 클릭
    time.sleep(1)  # 로그인 되는 시간을 위해 슬립
    driver.find_element(by='xpath', value='//*[@id="new.dontsave"]').click() #
    driver.find_element(by='xpath', value='//*[@id="HOME_SHORTCUT"]/ul/li[7]/a/div/picture/img').click()  # 블로그 아이콘 클릭


#블로그 링크따기
def get_blog(driver):
    for i in range(0, 30, 1):  # 반복 횟수로 불러올 게시글량 조절 가능
        driver.find_element(by='tag name', value='body').send_keys(Keys.PAGE_DOWN)
        time.sleep(0.5)
    html = driver.page_source  # 링크를 따와야 하므로 게시글이 많이 불러진 상태에서 html 객체를 생성
    soup = BeautifulSoup(html, 'html.parser')
    link_list = soup.select('.thumb_area__IeDdQ > a')   #링크 추출
    return link_list


#포스팅별 방문하여 좋아요 클릭
def click_like(driver, link_list):
    like_cnt = 0  # 좋아요 버튼 누른 횟수 저장 변수

    #추출된 링크 수 만큼 방문하여 좋아요 버튼 클릭
    for i in range(len(link_list)):
        site = (link_list[i]['href'])
        driver.get(site)
        time.sleep(0.5)
        try:
            is_like = driver.find_element(by='xpath', value='//*[@id="body"]/div[10]/div/div[1]/div/div/a').get_attribute('aria-pressed')   #좋아요 버튼 상태 확인
            #print(is_like)
        except Exception:   #간혹 공감 버튼 자체가 없는 게시글이 존재함
            print('공감 버튼 없음')
            continue
        if is_like == 'false':  #좋아요 버튼 상태가 안눌러져있는 상태일 경우에만 좋아요 버튼 클릭
            like_cnt += 1   #좋아요 횟수 1증가
            driver.find_element(by='xpath', value='//*[@id="body"]/div[10]/div/div[1]/div/div/a/span').click()  #하트 클릭
            time.sleep(0.5)
        try:
            time.sleep(1)
            alert = Alert(driver)   #팝업창으로 메시지 뜰 경우를 대비
            alert.accept()
        except Exception:
            continue
    return like_cnt


#크롬 닫기
def close(driver, link_list, like_cnt):
    print('총 {}개의 게시글 중 {}개의 게시글에 좋아요 버튼을 눌렀습니다.'.format(len(link_list), like_cnt))
    time.sleep(3)
    driver.close()


#메인
if __name__ == '__main__':
    driver = exec_chrom()
    login(driver)
    link_list = get_blog(driver)
    like_cnt = click_like(driver, link_list)
    close(driver, link_list, like_cnt)