import requests
import time


def Login(id,pw):
    Login_param = {
        'id': str(id),
        'pwd': str(pw)
    }
    Login_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    response = requests.post("http://ic.smartjoeun.co.kr/rtMember/index00_Proc.html", data=Login_param,
                             headers=Login_header, verify=False)  # , proxies='https://127.0.0.1')
    session = response.headers.get('Set-Cookie')

    print(response)
    print("Cookie:",session)
    return session


def next(session, i, j,id):
    next_param = {
        'log_num': '30891',
        'ch_num': '2722',
        'sch_num': str(i),
        'lec_num2': '2880',
        'lec_num': '2880',
        'lec_class_num': '642',
        'kiganDate': '2022-11-12',
        'lec_code': '202281213',
        'studyYNFlag': 'Y',
        'gpage': str(j),
        'mem_id':str(id),
        'eval_cd': '01',
        'kvalue': '4',
        'captcha': '',
        'otpval': '1',
        'url': '%2Fvod%2Fgong%2F346%2Findex.html',
        'orcs_yn_fg': '',
        'MAC_LOGIN': '',
        'ip_login': '',
        'CaptchaUrl': '',
        'k_count': '1',
        'emon_id': 'lms105a001',
        'timechk': ''
    }
    next_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'http://ic.smartjoeun.co.kr/rtSub07/index00.html?log_num=30889&chk=0',
        'Cookie': session
    }

    response = requests.post('http://ic.smartjoeun.co.kr/player/progress_start.asp', data=next_param,
                             headers=next_header, verify=False)  # , proxies=proxies)

    print(response)


def skip(session, i, j,id):
    skip_param = {
        # 'settime' : '3000',
        'log_num': '30891',
        'ch_num': '2722',
        'sch_num': str(i),  # '31695', ###################매번 바껴야함
        'lec_num2': '2880',
        'lec_num': '2880',
        'lec_class_num': '642',
        'kiganDate': '2022 - 11 - 12', ########################날짜 바꿔야하나?????????????????
        'lec_code': '202281213',
        'studyYNFlag': 'Y',
        'gpage': str(j),  # '8', ############################매번 바껴야함
        'mem_id': str(id),
        'eval_cd': '01',
        'kvalue': '1',
        'captcha': '',
        'otpval': '1',
        'url': '%2Fvod%2Fgong%2F346%2Findex.html',
        # 'url': '/vod/gong/346/index.html',
        'orcs_yn_fg': '',
        'MAC_LOGIN': '',
        'ip_login': '',
        'CaptchaUrl': '',
        'k_count': '0',
        'emon_id': 'lms105a001',
        'timechk': ''
    }
    skip_header = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Referer': 'http://ic.smartjoeun.co.kr/rtSub07/index00.html?log_num=30888&chk=0',
        'Cookie': session
    }
    # response = requests.post('http://ic.smartjoeun.co.kr:80', data=skip_param, headers=skip_header, verify=False)  # , proxies=cfg.proxies)
    response = requests.post('http://ic.smartjoeun.co.kr/player/progress_settime.asp?settime=3000', data=skip_param,
                             headers=skip_header, verify=False)  # , proxies=cfg.proxies)
    # session = response.request.headers.get('Cookie')
    # session = response.text
    # print(session)
    print('강의 번호:',i,' 페이지 번호:',j,' 완료')
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
    id = input('ID :')
    pw = input('비밀번호:')
    session = Login(id, pw)

    #강의번호 : prgress_star.asp 안에  sch_num 값  (예: 소셜네트워크분석 7차시 강의번혼는 31699)
    #페이지수 : 강의 총 페이지 수
    lecture= input('강의 번호:')
    page = input('총 페이지 수:')
    for j in range(int(page)):
        next(session, int(lecture), j + 1,id)
        skip(session, int(lecture) ,j + 1,id)
        time.sleep(0.1)

    # lec_num(session)