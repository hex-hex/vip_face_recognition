import cv2
import face_recognition
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp


def draw_multiline_line(img, point_list, color, thickness):
    line_list = zip(point_list[0:-1], point_list[1:])
    for line in line_list:
        cv2.line(img, line[0], line[1], color, thickness)


def draw_points(img, point_list, color, radius):
    for point in point_list:
        cv2.circle(img, point, radius, color, radius)


def main():
    inv_param = np.linalg.inv(np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17, 4)]))
    beautify_param = []

    for i in range(10):
        scale_1 = i * 0.05
        scale_2 = i * 0.01
        current_param = inv_param.dot(np.array((0, scale_1, scale_2, scale_1, 0)))
        b_param = np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17)]).dot(current_param)
        b_param = np.tile(b_param, (2, 1)).transpose()
        beautify_param.append(b_param)

    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        input_img = frame.copy() / 255.0
        height, width, depth = frame.shape
        src = np.array(
            ((0, 0), (width / 2, 0), (width - 1, 0), (0, height / 2), (0, height - 1), (width - 1, height - 1)))
        face_landmarks_list = face_recognition.face_landmarks(frame)
        transform = []
        dst = []
        for i in range(10):
            transform.append(PiecewiseAffineTransform())
            dst.append(np.array(
                ((0, 0), (width / 2, 0), (width - 1, 0), (0, height / 2), (0, height - 1), (width - 1, height - 1))))
        for face_landmarks in face_landmarks_list:
            chin = face_landmarks['chin']
            nose_bridge = face_landmarks['nose_bridge']

            # draw_multiline_line(frame, chin, (0, 255, 0), 2)
            # draw_multiline_line(input_img, chin, (0, 1.0, 0), 2)
            src = np.vstack([src, np.array(chin)])
            src = np.vstack([src, np.array(nose_bridge)])

            nose_point = face_landmarks['nose_bridge'][-1]
            for i in range(10):
                dst[i] = np.vstack(
                    [dst[i],
                     (np.array(chin) - np.tile(np.array(nose_point), (17, 1))) * beautify_param[i] + np.array(chin)])
                dst[i] = np.vstack([dst[i], np.array(nose_bridge)])

            # draw_multiline_line(frame,nose_bridge,(255,0,0),5)
            # nose_tip = face_landmarks['nose_tip']
            # draw_multiline_line(frame,nose_tip,(0,0,255),5)
        out_img = []
        for i in range(10):
            transform[i].estimate(src, dst[i])
            out_img.append(warp(frame, transform[i]))

        result = np.vstack([np.hstack([out_img[0], out_img[1]]),
                            np.hstack([out_img[2], out_img[3]]),
                            np.hstack([out_img[4], out_img[5]]),
                            np.hstack([out_img[6], out_img[7]]),
                            np.hstack([out_img[8], out_img[9]])]) * 255
        result = np.uint8(result)
        cv2.imshow('frame', result)
        cv2.imwrite('result.png', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
