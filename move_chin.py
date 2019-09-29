import cv2
import numpy as np
import face_recognition


def draw_multiline_line(img, point_list, color, thickness):
    line_list = zip(point_list[0:-1], point_list[1:])
    for line in line_list:
        cv2.line(img, line[0], line[1], color, thickness)


def draw_lines(img, point_list_1, point_list_2, color, thickness):
    line_list = zip(point_list_1, point_list_2)
    for line in line_list:
        cv2.line(img, line[0], line[1], color, thickness)


def draw_points(img, point_list, color, radius):
    for point in point_list:
        cv2.circle(img, point, radius, color, radius + 2)


def main():
    inv_param = np.linalg.inv(np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17, 4)]))
    param = inv_param.dot(np.array((0, 0.3, 0.02, 0.3, 0)))
    beautify_param = np.array([(i, i ** 2, i ** 3, i ** 4, 1) for i in range(0, 17)]).dot(param)
    beautify_param = np.tile(beautify_param, (2, 1)).transpose()
    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmarks in face_landmarks_list:
            chin = face_landmarks['chin']
            draw_points(frame, chin, (0, 255, 0), 2)
            nose_point = face_landmarks['nose_bridge'][-2]
            offset_chin = np.array(chin) - (np.array(chin) - np.tile(np.array(nose_point), (17, 1))) * beautify_param
            offset_chin = np.int32(offset_chin)
            offset_chin = tuple(map(tuple, offset_chin))
            draw_lines(frame, chin, offset_chin, (255, 0, 0), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
