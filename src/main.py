import cv2 as cv
from cv2 import VideoWriter, VideoWriter_fourcc
import numpy as np

# TODO: some reason tqdm is no longer working with Gooey
# from tqdm import tqdm
from gooey import Gooey, GooeyParser


def make_video(images: list, path="./assets/heatmap.avi"):
    print("Saving video...")
    fps = 24
    # fourcc is a 4-byte code used to specify the video codec
    fourcc = VideoWriter_fourcc(*'MJPG')
    is_first_frame = True
    for i in range(0, len(images)):
        if is_first_frame:
            is_first_frame = False
            height, width = images[i].shape[:2]
            vw = VideoWriter(path, fourcc, float(fps), (width, height))
        vw.write(images[i])
        print(f"progress: {i}/{len(images)}")

    # tell video writer we are finished
    vw.release()

def process_video(video_path: str, debug: bool) -> list:
    heatmap_imgs = []
    background_subtractor = cv.createBackgroundSubtractorMOG2()
    capture = cv.VideoCapture(video_path)

    if not capture.isOpened():
        print('Unable to open: ' + video_path)
        exit(0)

    print("generating heatmap for video: " + video_path)

    capture_length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))

    is_first_frame = True
    for i in range(0, capture_length):
        _, frame = capture.read()
        if frame is None:
            break

        if is_first_frame:
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            is_first_frame = False

        foreground_mask = background_subtractor.apply(frame)

        # use threshold to remove noise from video
        threshold = 2
        maxValue = 2
        _, dist = cv.threshold(foreground_mask, threshold, maxValue, cv.THRESH_BINARY)

        accum_image = cv.add(accum_image, dist)

        color_image_video = cv.applyColorMap(accum_image, cv.COLORMAP_HOT)
        # overlay the color mapped image to the first frame
        heatmap_frame = cv.addWeighted(frame, 0.7, color_image_video, 0.7, 0)
        heatmap_imgs.append(heatmap_frame)

        # for debugging
        if debug == True:
            #get the frame number and write it on the current frame
            cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
            cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
            #show the current frame and the fg masks
            cv.imshow('Frame', frame)
            cv.imshow('FG Mask', foreground_mask)

        cv.imshow('Heatmap', heatmap_frame)

        keyboard = cv.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

        print(f"progress: {i}/{capture_length}")

    # cleanup
    capture.release()
    cv.destroyAllWindows()
    return heatmap_imgs

@Gooey(
    program_name='Heatmap generator',
    progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
    progress_expr="current / total * 100",
    hide_progress_msg=True,
    timing_options={
        'show_time_remaining':True,
        'hide_time_remaining_on_complete':True
    })
def main():
    # drop in replacement for argparse.ArgumentParser
    parser = GooeyParser(
        description='Generates a heatmap for a video',
    )
    parser.add_argument('--path', action="store", required=True, help="The path to the file that you wish to generate the heatmap for", default="./assets/video.mp4", widget="FileChooser")
    parser.add_argument('--debug', action="store_true", default=False, help="Display additional debug information")
    parser.add_argument('--out', action="store", help="File path to save the heatmap to", type=str)


    results = parser.parse_args()

    video_path = results.path
    debug_mode=results.debug
    output_path = results.out
    heatmap_imgs = process_video(video_path, debug_mode)

    if output_path is not None:
        make_video(heatmap_imgs, output_path)

    print("finished")

if __name__ == '__main__':
    main()