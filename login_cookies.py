import requests
# import math
# import json
# from datetime import datetime, timedelta, timezone
#
# form_data = {
#     "login": "cmtest002",
#     "passwd": "Heaven@4394"
# }
#
# session = requests.session()
#
# r1 = session.post("https://demo-web-uat.paradise-soft.com.tw/apis/m/session", data=form_data)
#
#
# deposit_time = datetime.now().strftime('%Y-%m-%d %H:%M')
# form1_data = {
#     'transferamount': '10000',
#     'cardname': '微信',
#     'transfermethod': 'qrcode',
#     'cardbankcode': 'WEIXIN',
#     'cardnumber': '12345',
#     'transfertime': deposit_time,
#     'accountnumber': '00000_bot20210317161734_2125'
# }
#
# deposit_url = "https://demo-web-uat.paradise-soft.com.tw/apis/deposit"
# test = session.post(deposit_url, data=form1_data)
#
# start_time = f"{datetime.now().strftime('%Y-%m-%d')} 00:00"
# end_time = f"{(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')} 00:00"
# now_timestamp = math.floor((datetime.utcnow().replace(tzinfo=timezone.utc)).astimezone(timezone(timedelta(hours=8))).timestamp()*1000)
# deposit_type = 'deposit'
# # url = "https://demo-web-uat.paradise-soft.com.tw/m"
# url = f'https://demo-web-uat.paradise-soft.com.tw/apis/recharge/record?time=d&type={deposit_type}&starttime={start_time}&endtime={end_time}&pi=1&_={now_timestamp}&ps=25'
# res = session.get(url)
# tt = res.history[0].content.decode('utf-8').replace("'", "'")
#
#
# data = json.loads(tt)
# 
#
#
# pass

# import subprocess
# test = subprocess.call("adb devices", shell=True)
# pass

url ='https://demo-web-uat.paradise-soft.com.tw/m/'
session = requests.session()
#登入
form_data = {
'login':'cmtest001',
'passwd':'Heaven@4394'}

headers = {"user-agent": 'Mozilla/5.0'}

login_url = f'{url[:-2]}apis/{url[-2:]}session'
login = session.post(login_url, data=form_data)
cookies = session.cookies[".xy-web"]

# 轉帳
form_data = {
'from':'cp',
'to':'sp365',
'amount':'100'}

transfer_url = f'{url[:-2]}apis/my/wallet/transfer'
transfer = session.post(transfer_url, data=form_data)

# 進入遊戲
gameid = '3513'
enabletls = '0'
mobile = '1'
domain = 'demo-web-uat.paradise-soft.com.tw'

forward_game_url = f'{url[:-2]}apis/game/sp365_sport/forward-game?gameid={gameid}&enabletls={enabletls}&mobile={mobile}&domain={domain}'
forward_game = session.get(forward_game_url, headers=headers)
print(forward_game)