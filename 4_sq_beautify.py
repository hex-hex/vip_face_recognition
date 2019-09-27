import cv2
import face_recognition
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp


def draw_multiline_line(img, point_list, color, thickness):
    line_list = zip(point_list[0:-1], point_list[1:])
    for line in line_list:
        cv2.line(img, line[0], line[1], color, thickness)


def main():
    inv_param = np.linalg.inv(np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17, 4)]))
    param = inv_param.dot(np.array((0, 0.1, 0.02, 0.1, 0)))
    beautify_param = np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17)]).dot(param)
    beautify_param = np.tile(beautify_param, (2, 1)).transpose()
    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        input_img = frame.copy() / 255.0
        height, width, depth = frame.shape
        src = np.array(
            ((0, 0), (width / 2, 0), (width - 1, 0), (0, height / 2), (0, height - 1), (width - 1, height - 1)))
        dst = np.array(
            ((0, 0), (width / 2, 0), (width - 1, 0), (0, height / 2), (0, height - 1), (width - 1, height - 1)))
        face_landmarks_list = face_recognition.face_landmarks(frame)
        transform = PiecewiseAffineTransform()
        for face_landmarks in face_landmarks_list:
            chin = face_landmarks['chin']

            draw_multiline_line(frame, chin, (0, 255, 0), 2)
            draw_multiline_line(input_img, chin, (0, 1.0, 0), 2)
            src = np.vstack([src, np.array(chin)])

            nose_point = face_landmarks['nose_bridge'][-1]
            dst = np.vstack(
                [dst, (np.array(chin) - np.tile(np.array(nose_point), (17, 1))) * beautify_param + np.array(chin)])

            # draw_multiline_line(frame,nose_bridge,(255,0,0),5)
            # nose_tip = face_landmarks['nose_tip']
            # draw_multiline_line(frame,nose_tip,(0,0,255),5)
        transform.estimate(src, dst)
        out_img = warp(frame, transform)
        cv2.imshow('frame', np.hstack([input_img, out_img]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()