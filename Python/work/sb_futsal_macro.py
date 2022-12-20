import config as cfg
import calendar
import datetime
import requests
from fake_useragent import UserAgent
import time
from requests_toolbelt import MultipartEncoder

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':DES-CBC3-SHA'
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


def Date():
    curr_year = datetime.datetime.strftime(datetime.datetime.now(), '%Y')
    curr_year_int = int(curr_year)
    curr_month = datetime.datetime.strftime(datetime.datetime.now(), '%m')
    next_month_int = int(curr_month) + 1
    if next_month_int > 12:
        next_month_int = next_month_int - 12
        curr_year_int = curr_year_int + 1
    next_month = str(next_month_int)

    date_count = calendar.monthrange(curr_year_int, next_month_int)

    next_rsrvDt_arr = []
    next_rsrvDt_arr_dash =[]

    for reserve_day in range(1, date_count[1] + 1):
        weekday = calendar.weekday(curr_year_int, next_month_int, reserve_day)  # next_month_int, reserve_day)
        reserve_day_str = str(reserve_day)

        if weekday == calendar.THURSDAY:
            if len(next_month) == 1:
                next_month = next_month.zfill(2)
            if len(reserve_day_str) == 1:
                reserve_day_str = reserve_day_str.zfill(2)
            next_rsrvDt = str(curr_year_int) + next_month + reserve_day_str
            next_rsrvDt_dash = str(curr_year_int)+'-'+next_month+'-'+reserve_day_str
            next_rsrvDt_arr.append(next_rsrvDt)
            next_rsrvDt_arr_dash.append(next_rsrvDt_dash)

    print(next_rsrvDt_arr, next_rsrvDt_arr_dash)
    return next_rsrvDt_arr, next_rsrvDt_arr_dash

Date()
def Login():
    '''
    cookie = input()
    session='JSESSIONID='+cookie
    print(session)
    return session
    '''
    Login_param = {
        'password': cfg.login_PW_HASH,
        'username': cfg.login_ID
    }
    Login_header = {
        'User-Agent': useragent,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Connection': 'close'
    }

    response = requests.post(cfg.Login, data=Login_param, headers=Login_header, verify=False)  # , proxies=cfg.proxies)
    session = response.request.headers.get('Cookie')
    print(session)

    return session


def selectRentInfo(i, session):
    Reserve_Trigger = False
    while not Reserve_Trigger:
        selectRentInfo_param = MultipartEncoder(fields={
            'centerId': '01',
            'placeCode': '23',
            'eventCode': '64',
            'playCode': '01',
            'reserveDate': str(next_rsrvDt_arr[i]),
            'startTime': str(cfg.startTime)
        })
        selectRentInfo_header = {
            'Cookie': session,
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
            'Referer': 'https://www.gongdan.go.kr/ezPay/reservation/selectRentList.do',
            'Content-Type': selectRentInfo_param.content_type,
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'Sec-Ch-Ua': '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"macOS"',
            'Origin': 'https://www.gongdan.go.kr',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Sec-Fetch-Site': 'same-origin',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-User': '?1',
            'Sec-Fetch-Dest': 'document',
            'Accept-Encoding': 'gzip, deflate',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'close'
        }

        response = requests.post(cfg.selectRentInfo, data=selectRentInfo_param, headers=selectRentInfo_header, verify=False)#,proxies=cfg.proxies)
        status_code = str(response.history)

        print(status_code)
        return status_code
    '''
        while '302' in status_code:
            print('대기중:' + next_rsrvDt_arr[i], datetime.datetime.now())
            selectRentInfo(i, session)
            if '302' not in status_code:
                print('예약 성공:'+next_rsrvDt_arr[i], datetime.datetime.now())
                break


  
        if '200' not in status_code:
            print('예약 불가', next_rsrvDt_arr[i], datetime.datetime.now())
            selectRentInfo(i, session)
        else:
            print('예약 완료', next_rsrvDt_arr[i], datetime.datetime.now())
            Reserve_Trigger = True
        '''



def registerRent(i, session):
    registerRent_param = MultipartEncoder(fields={
        'centerId': '01',
        'placeCode': '23',
        'eventCode': '64',
        'playCode': '01',
        'reserveDate': str(next_rsrvDt_arr[i]),
        'startTime': str(cfg.startTime),
        'endTime': str(cfg.endTime),
        'playName': '풋살1코트',
        'discount_code': '00001',
        'trs_type': '00',
        'payment_method': '02',
        'cash_amount': '0',
        'card_amount': '0',
        'unit_amount': '72000',
        'discount_amount': '0',
        'total_amount': '72000',
        'qty': '1',
        'deal_record': '대관접수 : 풋살1코트 ('+str(next_rsrvDt_arr_dash[i])+' 10:00)',#변수
        'print_desc_1': '대관료 : 풋살1코트',
        'print_desc_2': '('+ str(next_rsrvDt_arr_dash[i]) +' 10:00 ~ 12:00)',#변수
        'rentTitle_finish': '풋살1코트',
        'reserveDate_finish': str(next_rsrvDt_arr[i]),
        'startTime_finish': str(cfg.startTime),
        'endTime_finish': str(cfg.endTime),
        'haengsaName': '풋살',
        'group_code': '00',
        'dancheName': '개인대관',
        'daeguanObject': '풋살',
        'participants': '15'
    })

    registerRent_header = {
        'Cookie': session,
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
        'Content-Type': registerRent_param.content_type,
        'Referer': 'https://www.gongdan.go.kr/ezPay/reservation/selectRentInfo.do',
        'Sec-Ch-Ua': '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With': 'XMLHttpRequest',
        'Ajax': 'true',
        'Sec-Ch-Ua-Platform': '"macOS"',
        'Origin': 'https://www.gongdan.go.kr',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Dest': 'empty',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Connection': 'close'
    }

    response = requests.post(cfg.registerRent, data=registerRent_param, headers=registerRent_header,
                             verify=False).json()  # , proxies=cfg.proxies
    nextPcUrl = response.get('responseMap').get('data').get('nextPcUrl')

    return nextPcUrl


if __name__ == "__main__":
    ua = UserAgent(verify_ssl=False)
    useragent = ua.chrome

    file_path = cfg.macbook_file_path

    #next_rsrvDt_arr, next_rsrvDt_arr_dash = Date()
    next_rsrvDt_arr = ['20220725','20220726','20220727','20220728']
    next_rsrvDt_arr_dash = ['2022-07-25','2022-07-26','2022-07-27','2022-07-28']

    session = Login()

    Trigger = False
    while not Trigger:
        executeTime = datetime.datetime.now()
        if executeTime.hour == 16 and executeTime.minute == 45 and executeTime.second == 59:
            for i in range(len(next_rsrvDt_arr)):
                status_code = '302'
                while '302' in status_code:
                    status_code = selectRentInfo(i, session)

                    if '302' not in status_code:
                        print('예약 성공:' + next_rsrvDt_arr[i], datetime.datetime.now())
                        break

            for i in range(len(next_rsrvDt_arr)):
                nextPcUrl = registerRent(i, session)
                file_name = str(next_rsrvDt_arr[i]) + ".txt"
                file =file_path+file_name
                with open(file, "w", encoding="utf-8") as f:
                    f.write(nextPcUrl)
                print('결제페이지:' + nextPcUrl, datetime.datetime.now())
            Trigger = True
        else:
            time.sleep(1)
