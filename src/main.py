import cv2 as cv
from cv2 import VideoWriter, VideoWriter_fourcc
import copy
import numpy as np

def make_video(images):
    fps = 24
    print("creating heatmap video...")
    # fourcc is a 4-byte code used to specify the video codec
    fourcc = VideoWriter_fourcc(*'MJPG')
    isFirstFrame = True
    print(len(images))
    for (frame) in images:
        if isFirstFrame:
            isFirstFrame = False
            height, width = frame.shape[:2]
            vw = VideoWriter('./assets/heatmap.avi', fourcc, float(fps), (width, height))
        vw.write(frame)

    # tell video writer we are finished
    vw.release()

    pass

def main():
    heatmap_imgs = []
    background_subtractor = cv.createBackgroundSubtractorMOG2()
    videoPath = "assets/video.mp4"
    capture = cv.VideoCapture(videoPath)

    if not capture.isOpened():
        print('Unable to open: ' + videoPath)
        exit(0)

    print("generating heatmap for video: " + videoPath)

    isFirstFrame = True
    while True:
        ret, frame = capture.read()
        if frame is None:
            break

        if isFirstFrame:
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            isFirstFrame = False

        foreground_mask = background_subtractor.apply(frame)

        # for debugging
        #get the frame number and write it on the current frame
        cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))

        # use threshold to remove noise from video
        threshold = 2
        maxValue = 2
        ret, dist = cv.threshold(foreground_mask, threshold, maxValue, cv.THRESH_BINARY)

        accum_image = cv.add(accum_image, dist)

        color_image_video = cv.applyColorMap(accum_image, cv.COLORMAP_HOT)
        # overlay the color mapped image to the first frame
        heatmap_frame = cv.addWeighted(frame, 0.7, color_image_video, 0.7, 0)
        heatmap_imgs.append(heatmap_frame)

        #show the current frame and the fg masks
        cv.imshow('Frame', frame)
        cv.imshow('FG Mask', foreground_mask)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    # cleanup
    capture.release()
    cv.destroyAllWindows()
    make_video(heatmap_imgs)

    print("finished")

if __name__ == '__main__':
    main()