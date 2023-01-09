import os

from datetime import datetime
from airtest.core.api import *
from airtest.core.api import G
from airtest.core.api import connect_device
from airtest.core.settings import Settings as ST

def deviceConnect(uuid=None):
    if uuid == None:
        uuid = "1576457605007R5"
    device = connect_device("Android://127.0.0.1:5037/%s?cap_method=minicap&touch_method=adb" % (uuid))

    return device

if __name__ == '__main__':
    device = deviceConnect()
    pass