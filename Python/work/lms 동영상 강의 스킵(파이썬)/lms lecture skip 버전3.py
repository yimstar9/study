
import requests
import time
import json
from bs4 import BeautifulSoup
import re
import sys

def Login(i,p):
    #proxies = {'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'}

    Login_param = {
        'id' : i,
        'pwd' : p,
        'mac_add' : '',
        'on_de_chk' : '1',
        'x' : '35',
        'y' : '46'
    }
    Login_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    response = requests.post('http://ic.smartjoeun.co.kr/rtMember/index00_Proc.html', data=Login_param, headers=Login_header, verify=False)
    session = response.headers.get('Set-Cookie')

    print(response.ok)

    return session

def next(session,id, j,**args):

    next_param = {
        'log_num' : args['log_num'], ################값이 바뀌는듯
        'ch_num' : args['ch_num'], ################값이 바뀌는듯
        'sch_num' : args['sch_num'],
        'lec_num2' : args['lec_num'], ################값이 바뀌는듯
        'lec_num' : args['lec_num'], ################값이 바뀌는듯
        'lec_class_num' : args['lec_class_num'], ################값이 바뀌는듯
        'kiganDate' : args['kiganDate'],
        'lec_code' : args['lec_code'],
        'studyYNFlag' : args['studyYNFlag'],
        'gpage' : str(j),
        'mem_id' : id,
        'eval_cd' : '01',
        'kvalue' : args['kvalue'],
        'captcha' : '',
        'otpval' : '1',
        'url' : args['url'],
        'orcs_yn_fg' : '',
        'MAC_LOGIN' : '',
        'ip_login' : '',
        'CaptchaUrl' : '',
        'k_count' : '1',
        'emon_id' : 'lms105a001',
        'timechk' : ''
    }
    next_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'http://ic.smartjoeun.co.kr/rtSub07/index00.html?log_num=30889&chk=0',
        'Cookie': session
    }

    response = requests.post('http://ic.smartjoeun.co.kr/player/progress_start.asp', data=next_param, headers=next_header, verify=False)#, proxies=proxies)
    response1 = requests.get('http://ic.smartjoeun.co.kr/player/progress_start.asp')
    response1.text

def extract(a):
    b=re.findall('showLectureLayer_ooic'+'\(([^)]+)', a)
    d=list(b[len(b)-1].replace('\'','').replace(' ','').split(','))
    c=['url', 'log_num', 'ch_num', 'sch_num', 'lec_num', 'kiganDate', 'lec_code', 'studyYNFlag', 'orcs_yn_fg',
       'MAC_LOGIN','ip_login', 'CaptchaUrl', 'k_count', 'kvalue', 'lec_class_num', 'site_key']
    param_dict = dict(zip(c,d))

    return param_dict


def info(session,num):     #########수강 클릭 하면 받아오는 정보
    info_param={
        'log_num' : str(num),
        'chk' : '0'
    }
    info_headers = {
        'Content-Type' : 'text/html; Charset=euc-kr',
        'Cookie' : session
    }
    response = requests.get('http://ic.smartjoeun.co.kr/rtSub07/index00.html', params=info_param, headers=info_headers)
    html = response.text
    #print(html)
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find_all('a',class_='buttonNEW2 Burgendiborder fwhite')

    return title

def page(session,i,**args):

    page_param={
        'slea_pageNum' : str(i+1),
        'sch_num' : args['sch_num'],
        'chul_suk_page' : str(i)
    }
    page_header={
        'Referer':'http://ic.smartjoeun.co.kr/vod/gong/137/index.html',
        'Cookie': session
    }

    response = requests.get('http://ic.smartjoeun.co.kr/vod/gong/137/index.html', params=page_param, headers=page_header)
    return response.ok

def skip(session,id, j,**args):
    skip_param = {
        'settime' : '3000', ####강의 각 단원별 넘길 시간
        'log_num' : args['log_num'], ################값이 바뀌는듯
        'ch_num' : args['ch_num'], ################값이 바뀌는듯
        'sch_num' : args['sch_num'], #'31701', ###################매번 바껴야함
        'lec_num2' : args['lec_num'], ################값이 바뀌는듯
        'lec_num' : args['lec_num'], ################값이 바뀌는듯
        'lec_class_num' : args['lec_class_num'],
        'kiganDate' : args['kiganDate'],
        'lec_code' : args['lec_code'],
        'studyYNFlag' : args['studyYNFlag'],
        'gpage': str(j), #'8', ############################매번 바껴야함
        'mem_id' : id,
        'eval_cd' : '01',
        'kvalue' : args['kvalue'],
        'captcha' :'',
        'otpval' : '1',
        'url' : args['url'], ################값이 바뀌는듯
        #'url': '/vod/gong/346/index.html',
        'orcs_yn_fg' :'',
        'MAC_LOGIN' :'',
        'ip_login' :'',
        'CaptchaUrl' :'',
        'k_count' : '0',
        'emon_id' : 'lms105a001',
        'timechk' :''
    }
    skip_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'http://ic.smartjoeun.co.kr/rtSub07/index00.html',
        'Cookie': session
    }

    response = requests.post('http://ic.smartjoeun.co.kr/player/progress_settime.asp', data=skip_param, headers=skip_header, verify=False)  # , proxies=cfg.proxies)
    #session = response.request.headers.get('Cookie')
    #session = response.text
    #print(session)
    print(j,"페이지 완료")
    return


if __name__ == "__main__":
    ID = 'yimstar9'
    PW = 'dlatjdrn1!'
    session = Login(ID,PW)

    txt=info(session,30887) ####무슨강의 들을건지 숫자 자리에 번호 넣어줘야함
    if not txt:             ####강의 목록에서 수강버튼 post 보내면 log_num parameter값
        print("로그인 에러")  ####log_num이
        sys.exit()

    key=extract(str(txt)) #강의정보 받아옴
    print(key)

    for i in range(30): ##동영상 페이지 정보 받아옴
        gp=page(session,i+1,**key)
        if not gp:
            gpage=i+1
            print("총페이지:",gpage)
            break


    for j in range(gpage): #강의 스킵부분
         next(session, ID, j+1,**key)
         skip(session, ID, j+1,**key)
         time.sleep(0.1)


