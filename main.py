import cv2
import face_recognition


def main():
    cv2.namedWindow("frame")
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        face_landmarks_list = face_recognition.face_landmarks(frame)
        for face_landmarks in face_landmarks_list:
            chin = face_landmarks['chin']
            chin_line_list = zip(chin[0:-1], chin[1:])
            for chin_line in chin_line_list:
                cv2.line(frame, chin_line[0], chin_line[1], (0, 255, 0), 5)
            # face_landmarks['nose_bridge']
            # face_landmarks['nose_tip']
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
