import cv2
import dlib

model = dlib.simple_object_detector('../resources/model.svm')


class Detector:
        
    def main(self):

        capture = cv2.VideoCapture(0)

        while True:
            _, frame = capture.read()
            
            detecteds_objects = model(frame)

            for d in detecteds_objects:
                l, t, r, b = (
                    int(d.left()),
                    int(d.top()),
                    int(d.right()),
                    int(d.bottom())
                )

                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 255), 2)
                cv2.rectangle(frame, (l, b - 35), (r, b), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, 'Keyboard', (l + 6, b - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

            cv2.imshow('Custom Objects', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Detector().main()
