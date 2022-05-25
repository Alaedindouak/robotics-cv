import cv2 as cv
import numpy as np

# Divide an image in patches and saves them to the disk
def devide_image_to_patches(image):
    for i in range(3):
        for j in range(3):
            img_patche = image[(img.shape[0] // 3 * i):(img.shape[0] // 3 * (i + 1)),
                (img.shape[1] // 3 * j):(img.shape[1] // 3 * (j + 1))]

            cv.imwrite(f'output/img_patche_{str(i)}_{str(j)}.png', img_patche)


# Warp an image with a mouse clicking
def warp_image(image):
    points_axis = np.zeros((4, 2), np.int)
    counter = 0
    
    def mouse_point(event, x, y, flags, params):
        nonlocal counter

        if event == cv.EVENT_LBUTTONDOWN:
            points_axis[counter] = x, y
            counter += 1

    while True:
        if counter == 4:
            width, height = 250, 350
            pts1 = np.float32([points_axis[0], points_axis[1], points_axis[2], points_axis[3]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
            matrix = cv.getPerspectiveTransform(pts1, pts2)
            output = cv.warpPerspective(image, matrix, (width, height))
            cv.imshow('Warp image', output)

        for x in range(4):
            center_point = (points_axis[x][0], points_axis[x][1])
            green_color = (0, 255, 0)
            cv.circle(
                img=image,
                center=center_point,
                radius=5,
                color=green_color,
                thickness=cv.FILLED
            )

        cv.imshow('Mr Bean', image)
        cv.setMouseCallback('Mr Bean', mouse_point)

        cv.waitKey(1)
        # cv.destroyAllWindows()


# Record a video from a webcam to a mp4-file with OpenCV 
def record_video_from_webcam():

    vid_capture = cv.VideoCapture(0)

    if not vid_capture.isOpened():
        print('error opening the video file')
    else:
        
        frame_width = int( vid_capture.get(cv.CAP_PROP_FRAME_WIDTH) )
        frame_height = int( vid_capture.get(cv.CAP_PROP_FRAME_HEIGHT) )
        frame_size = (frame_width, frame_height)
        fps = 30


        # second - used to compress the frame to mp4 
        vid_output = cv.VideoWriter('output/out.mp4', cv.VideoWriter_fourcc(*'XVID'), fps, frame_size)

        while vid_capture.isOpened():

            ret, frame = vid_capture.read()
            if ret:
                # write the frame to the output file
                vid_output.write(frame)

                cv.imshow('streaming...', frame)
                key = cv.waitKey(20)

                # 113 is ASCII code for q key
                if key == 113:
                    break
            else:
                print('stream disconnected')
                break

    vid_capture.release()
    vid_output.release()


if __name__ == "__main__":
    img = cv.imread('resources/mr__bean.png')
    
    # devide_image_to_patches(img)
    # warp_image(img)
    record_video_from_webcam()