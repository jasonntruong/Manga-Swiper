import cv2
import mediapipe

pollCounter = 0
oldcx = None
oldcy = None

capture = cv2.VideoCapture(0)

mediapipeHands = mediapipe.solutions.hands

myHands = mediapipeHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawHands = mediapipe.solutions.drawing_utils

while True:
    found, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    foundHands = myHands.process(imgRGB)

    if foundHands.multi_hand_landmarks:
        for hands in foundHands.multi_hand_landmarks:
            for id, location in enumerate(hands.landmark):
                h, w, c = img.shape
                cx = int(location.x * w)
                cy = int(location.y * h)
                cv2.circle(img, (cx, cy), 3, (255,0,255), cv2.FILLED)

                if pollCounter == 200:
                    pollCounter = 0
                    if oldcx and oldcy:
                        if abs(oldcx - cx) < 100 and (oldcy - cy) > 100:
                            print("SWIPE DETECTED")
                    
                    oldcx = cx
                    oldcy = cy
                pollCounter += 1
            drawHands.draw_landmarks(img, hands, mediapipeHands.HAND_CONNECTIONS)

            cv2.putText(img,str("X: "+str(cx)+" Y: "+str(cy)), (10,70), cv2.QT_FONT_NORMAL, 1, (0,0,255), 1)


    cv2.imshow("MangaReader", img)
    cv2.waitKey(1)