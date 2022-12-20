from operator import mul
import config as cfg
import datetime
import requests
from fake_useragent import UserAgent
import time
from requests_toolbelt import MultipartEncoder
import pandas as pd
import multiprocessing

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':DES-CBC3-SHA'
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


def Date():
    NextMonthBegin=pd.datetime.now().date()+pd.offsets.MonthBegin()
    NextMonthEnd=pd.datetime.now().date()+pd.offsets.MonthBegin()+pd.offsets.MonthEnd()
    start=NextMonthBegin.strftime('%Y%m%d')
    end=NextMonthEnd.strftime('%Y%m%d')

    dt_index = pd.date_range(start=start, end=end,freq='W-THU')
    next_rsrvDt_arr = dt_index.strftime("%Y%m%d").tolist()
    next_rsrvDt_arr_dash = dt_index.strftime("%Y-%m-%d").tolist()
    print(next_rsrvDt_arr,next_rsrvDt_arr_dash)

    return next_rsrvDt_arr, next_rsrvDt_arr_dash


class Reservation():
    def __init__(self, i):
        self.i = i

    def Login(self):
        print(cfg.login_ID[i])
        Login_param = {
            'password': cfg.login_PW_HASH[i],
            'username': cfg.login_ID[i]
        }
        Login_header = {
            'User-Agent': useragent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        response = requests.post(cfg.Login, data=Login_param, headers=Login_header, verify=False)  # , proxies=cfg.proxies)
        self.session = response.request.headers.get('Cookie')
        print(self.session)

        return self.session

    def Available(self):
        Trigger = False
        while not Trigger:
            executeTime = datetime.datetime.now()
            if executeTime.hour == 19 and executeTime.minute == 51 and executeTime.second == 15:
                status_code = '302'
                while '302' in status_code:
                    status_code = Reservation.selectRentInfo(self)
                    if '302' not in status_code:
                        break
                Trigger = True
            else:
                time.sleep(1)

    def selectRentInfo(self):
        print(next_rsrvDt_arr[i])
        selectRentInfo_param = MultipartEncoder(fields={
            'centerId': '01',
            'placeCode': '23',
            'eventCode': '64',
            'playCode': '01',
            'reserveDate': str(next_rsrvDt_arr[i]),
            'startTime': str(cfg.startTime)
        })
        selectRentInfo_header = {
            'Cookie': self.session,
            'User-Agent': useragent,
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
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        }

        response = requests.post(cfg.selectRentInfo, data=selectRentInfo_param, headers=selectRentInfo_header, verify=False)#,proxies=cfg.proxies)
        status_code = str(response.history)

        print(status_code)
        return status_code

    def registerRent(self):
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
            'Cookie': self.session,
            'User-Agent': useragent,
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
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7'
        }

        response = requests.post(cfg.registerRent, data=registerRent_param, headers=registerRent_header,
                                verify=False).json()  # , proxies=cfg.proxies
        nextPcUrl = response.get('responseMap').get('data').get('nextPcUrl')

        return nextPcUrl


if __name__ == "__main__":
    ua = UserAgent(verify_ssl=False)
    useragent = ua.chrome

    file_path = cfg.macbook_file_path

    next_rsrvDt_arr = ['20220725','20220726','20220727','20220728','20220729']
    next_rsrvDt_arr_dash = ['2022-07-25','2022-07-26','2022-07-27','2022-07-28','2022-07-29']
    #next_rsrvDt_arr, next_rsrvDt_arr_dash = Date()


    for i in range(0, len(next_rsrvDt_arr)):#4주 or 5주에 따라 클래스가 생성되야함.
        if i == 0:
            p0 = Reservation(i)
            p0_thread=multiprocessing.process(target=p0.Available())
            p0_thread.start()
        elif i == 1:
            p1 = Reservation(i)
            p1_thread = multiprocessing.process(target=p1.Available())
            p1_thread.start()
        elif i == 2:
            p2 = Reservation(i)
            p2_thread.start()
        elif i == 3:
            p3 = Reservation(i)
        else:
            p4 = Reservation(i)



    p0_thread.start()
    p1_thread.start()
    p2_thread.start()
    p3_thread.start()
    p4_thread.start()  
    p0_thread.join()
    p1_thread.join()
    p2_thread.join()
    p3_thread.join()
    p4_thread.join()

    p0.registerRent()
    p1.registerRent()
    p2.registerRent()
    p3.registerRent()
    p4.registerRent()
    







    '''
    2nd = Reservation(1)
    3rd = Reservation(2)
    4th = Reservation(3)
    5th = Reservation(4)
    
    1st_Process = multiprocessing.Process(target=1st.selectRentInfo)
    2nd_Process = multiprocessing.Process(target=2nd.selectRentInfo)
    3rd_Process = multiprocessing.Process(target=3rd.selectRentInfo)
    4th_Process = multiprocessing.Process(target=4th.selectRentInfo)
    5th_Process = multiprocessing.Process(target=5th.selectRentInfo)

    1st_Process.start()
    2nd_Process.start()
    3rd_Process.start()
    4th_Process.start()
    5th_Process.start()

    1st_Process.join()
    2nd_Process.join()
    3rd_Process.join()
    4th_Process.join()
    5th_Process.join()


    Trigger = False
    while not Trigger:
        executeTime = datetime.datetime.now()
        if executeTime.hour == 16 and executeTime.minute == 45 and executeTime.second == 59:
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
    '''