import cv2
import face_recognition


def draw_multiline_line(img, point_list, color, thickness):
    line_list = zip(point_list[0:-1], point_list[1:])
    for line in line_list:
        cv2.line(img, line[0], line[1], color, thickness)


def main():
    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CV_CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CV_CAP_PROP_FRAME_HEIGHT, 480)
    while (True):
        ret, frame = cap.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmarks in face_landmarks_list:
            chin = face_landmarks['chin']
            draw_multiline_line(frame, chin, (0, 255, 0), 5)

            nose_bridge = face_landmarks['nose_bridge']
            draw_multiline_line(frame, nose_bridge, (255, 0, 0), 5)

            nose_tip = face_landmarks['nose_tip']
            draw_multiline_line(frame, nose_tip, (0, 0, 255), 5)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
