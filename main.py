# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import subprocess
import time

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    status = subprocess.call("adb devices", shell=True)
    assert status == 0, "something is wrong"
    # subprocess.call("adb -s QV720QML24 shell", shell=True)
    # time.sleep(600)
    for i in range(15):
        subprocess.call("adb shell input tap 1574 929", shell=True)
        print("run: ", i+1, " times")
        time.sleep(600)

    # for i in range(7):
    #     subprocess.call("adb shell input tap 2330 990", shell=True)
    #     time.sleep(3)
    #     subprocess.call("adb shell input tap 2000 800", shell=True)
    #     print("run: ", i+1, " times")
    #     time.sleep(180)
    #     print("3min count")
    #     subprocess.call("adb shell input tap 2000 800", shell=True)
    #     time.sleep(5)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
