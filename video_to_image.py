import cv2
import os


video_path = r"C:\deepsea_testuse\snapshot_video\basic_item.mp4"
extract_folder = r"C:\Users\norman_cheng\Desktop\test_request\record_test001"
extract_frequency = 30

def extract_frames(video_path, dst_folder, index):
    video = cv2.VideoCapture(video_path)
    count = 1
    while True:
        rval, frame = video.read()
        if frame is None:
            break
        if count % extract_frequency == 0:
            fileName =  "{}/test{:>03d}.jpg".format(extract_folder, index)
            cv2.imwrite(fileName, frame)
            index += 1
        count += 1
    video.release()

    print("Total save {} pics".format(index-1))

if __name__ == '__main__':
    extract_frames(video_path, extract_folder, 1)
