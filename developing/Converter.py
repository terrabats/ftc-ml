import cv2
import os
import os
from os.path import isfile, join

count = 1

def imsToVideo(name):
    image_folder = 'out_ims'
    video_name = f'out_videos/{name}'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 20, (width,height))

    reoims = [None]*10000
    for i in range(len(images)):
        num = int(images[i].replace('.png', ''))
        reoims[num] = images[i]

    reoims = a = [x for x in reoims if x is not None]
    for im in reoims:
        video.write(cv2.imread(os.path.join(image_folder, im)))

    cv2.destroyAllWindows()
    video.release()
def videoToIms(name):
    global count
    sec = 0
    frameRate = 0.5
    success = getFrame(sec,name)
    while success:
        count += 1
        sec += frameRate
        sec = round(sec, 2)
        success = getFrame(sec,name)
def getFrame(sec,name):
    global count
    vidcap = cv2.VideoCapture(f'train_videos/{name}')
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite("train_ims/" + str(count) + ".png", image)
    return hasFrames

#imsToVideo('video_out.mp4')

vids = [v for v in os.listdir('train_videos/') if v.endswith(".mp4")]
for vid in vids:
    print(vid)
    videoToIms(vid)