import cv2 as cv


def frame_diff(prev_frame, cur_frame, next_frame):

    diff_frames1 = cv.absdiff(next_frame, cur_frame)

    diff_frames2 = cv.absdiff(cur_frame, prev_frame)
    return cv.bitwise_and(diff_frames1, diff_frames2)


def get_frame(cap):
    ret, frame = cap.read()

    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)

    return cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    scaling_factor = 1

    prev_frame = get_frame(cap)
    cur_frame = get_frame(cap)
    next_frame = get_frame(cap)

    while True:
        cv.imshow('Object Moviment', frame_diff(prev_frame, cur_frame, next_frame))

        prev_frame = cur_frame
        cur_frame = next_frame
        next_frame = get_frame(cap)
        key =  cv.waitKey(10) 
        if key == 27:
            break
    cv.destroyAllWindows()