
import requests
import time
import json
from bs4 import BeautifulSoup

# def Login():
#     Login_param = {
#         'id' : 'yimstar9',
#         'pwd' : 'dlatjdrn1!'
#     }
#     Login_header = {
#         'Content-Type': 'application/x-www-form-urlencoded',
#     }
#
#     response = requests.post("http://ic.smartjoeun.co.kr/rtMember/index00_Proc.html", data=Login_param, headers=Login_header, verify=False)#, proxies='https://127.0.0.1')
#     session = response.headers.get('Set-Cookie')
#
#     print(session)
#
#     return session
def Login():
    proxies = {'http': 'http://127.0.0.1:8080', 'https': 'http://127.0.0.1:8080'}

    Login_param = {
        'id' : 'yimstar9',
        'pwd' : 'dlatjdrn1!',
        'mac_add' : '',
        'on_de_chk' : '1',
        'x' : '35',
        'y' : '46'
    }
    Login_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    response = requests.post('http://ic.smartjoeun.co.kr/rtMember/index00_Proc.html', data=Login_param, headers=Login_header, verify=False, proxies=proxies)
    session = response.headers.get('Set-Cookie')

    print(session)
    return session

def next(session, i, j):

    next_param = {
        'log_num' : '30890', ################값이 바뀌는듯
        'ch_num' : '2723', ################값이 바뀌는듯
        'sch_num' : str(i),
        'lec_num2' : '2881', ################값이 바뀌는듯
        'lec_num' : '2881', ################값이 바뀌는듯
        'lec_class_num' : '642', ################값이 바뀌는듯
        'kiganDate' : '2022-11-12',
        'lec_code' : '202281213',
        'studyYNFlag' : 'Y',
        'gpage' : str(j),
        'mem_id' : 'yimstar9',
        'eval_cd' : '01',
        'kvalue' : '4',
        'captcha' : '',
        'otpval' : '1',
        'url' : '%2Fvod%2Fgong%2F237%2Findex.html',
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
    #print(type(response1.text))
    #dict = json.loads(response1.text)



# def info(session,x):
#     url = str(x)
#     info_cookie={
#         'Cookie' : session
#     }
#     response = requests.get(url,params=info_cookie)
#     html = response.text
#     print(html)
#     #soup = BeautifulSoup(html, 'html.parser')
#     #title = soup.find('a',class_='buttonNEW2 Burgendiborder fwhite')
#     return
def info(session):
    info_param={
        'log_num' : '30890',
        'chk' : '0',
    }
    info_headers = {
        'Content-Type' : 'text/html; Charset=euc-kr',
        'Cookie' : session
    }
    response = requests.get('http://ic.smartjoeun.co.kr/rtSub07/index00.html', params=info_param, headers=info_headers)
    html = response.text
    print(html)

def skip(session, i,j):
    skip_param = {
        #'settime' : '3000',
        'log_num' : '30890', ################값이 바뀌는듯
        'ch_num' : '2723', ################값이 바뀌는듯
        'sch_num' : str(i), #'31701', ###################매번 바껴야함
        'lec_num2' : '2881', ################값이 바뀌는듯
        'lec_num' : '2881', ################값이 바뀌는듯
        'lec_class_num' : '642',
        'kiganDate' : '2022 - 11 - 12',
        'lec_code' : '202281213',
        'studyYNFlag' : 'Y',
        'gpage': str(j), #'8', ############################매번 바껴야함
        'mem_id' : 'yimstar9',
        'eval_cd' : '01',
        'kvalue' : '1',
        'captcha' :'',
        'otpval' : '1',
        'url' : '%2Fvod%2Fgong%2F237%2Findex.html', ################값이 바뀌는듯
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
        'Referer': 'http://ic.smartjoeun.co.kr/rtSub07/index00.html?log_num=30888&chk=0',
        'Cookie': session
    }
    #response = requests.post('http://ic.smartjoeun.co.kr:80', data=skip_param, headers=skip_header, verify=False)  # , proxies=cfg.proxies)
    response = requests.post('http://ic.smartjoeun.co.kr/player/progress_settime.asp?settime=3000', data=skip_param, headers=skip_header, verify=False)  # , proxies=cfg.proxies)
    # session = response.request.headers.get('Cookie')
    #session = response.text
    #print(session)
    print(i,j)
    return

# def lec_num(session):
#     lec_num_param = {
#         'log_num' : '30889',
#         'chk' : '0'
#     }
#     lec_num_header = {
#         'Cookie': session
#     }
#     response = requests.get('http://ic.smartjoeun.co.kr/rtSub07/index00.html', params=lec_num_param, headers=lec_num_header, verify=False)
#     print(response.text)

if __name__ == "__main__":
    session = Login()
    info(session)
    for j in range(9):
        next(session, 31701, j+1)
        skip(session, 31701,j+1)
        time.sleep(0.1)



    #http: // ic.smartjoeun.co.kr / rtSub07 / index00.html?log_num = 30890 & chk = 0 주소로 request get 해서

    #showLectureLayer(url, log_num, ch_num, sch_num, lec_num, kiganDate, lec_code, studyYNFlag, orcs_yn_fg, MAC_LOGIN,ip_login, CaptchaUrl, k_count, kvalue, lec_class_num, site_key);

    # < a    # href = "#" #class ="buttonNEW2 Burgendiborder fwhite" onfocus="this.blur();" onclick="go_datecount('30890');
    # showLectureLayer_ooic('/vod/gong/237/index.html', '30890', '2723', '31702', '2881', '2022-11-12', '202281213', 'Y','','','','OTP_Ifame.asp','1','2', '642','100' ,'10' ,'0','0','S','2193','a001_license','lms105a001'  );" > 수강 < / a >
    #
    #http: // ic.smartjoeun.co.kr / rtSub07 / index00.html?log_num = 30890 & chk = 0
    # < / form >
    #
    # < script
    # type = 'text/javascript'
    # language = 'javascript' >
    #
    # function
    # showLectureLayer_hrd()
    # {
    #
    #     var
    # url = document.getElementById("url").value;
    # var
    # log_num = document.getElementById("log_num").value;
    # var
    # ch_num = document.getElementById("ch_num").value;
    # var
    # sch_num = document.getElementById("sch_num").value;
    #
    # var
    # lec_num = document.getElementById("lec_num").value;
    # var
    # kiganDate = document.getElementById("kiganDate").value;
    # var
    # lec_code = document.getElementById("lec_code").value;
    # var
    # studyYNFlag = document.getElementById("studyYNFlag").value;
    # var
    # orcs_yn_fg = document.getElementById("orcs_yn_fg").value;
    #
    # var
    # MAC_LOGIN = document.getElementById("MAC_LOGIN").value;
    # var
    # ip_login = document.getElementById("ip_login").value;
    # var
    # CaptchaUrl = document.getElementById("CaptchaUrl").value;
    #
    # var
    # k_count = document.getElementById("k_count").value;
    # var
    # kvalue = document.getElementById("kvalue").value; // 평가방법(ex.
    # '시험_1')
    # var
    # lec_class_num = document.getElementById("lec_class_num").value;
    # var
    # mem_id = document.getElementById("mem_id").value;
    # var
    # site_key = 'a001_license';
    #
    # showLectureLayer(url, log_num, ch_num, sch_num, lec_num, kiganDate, lec_code, studyYNFlag, orcs_yn_fg, MAC_LOGIN,
    #                  ip_login, CaptchaUrl, k_count, kvalue, lec_class_num, site_key);
    #

