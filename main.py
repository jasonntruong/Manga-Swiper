import cv2
import mediapipe
from pynput.keyboard import Key, Controller

pollCounter = 0
oldcx = None
oldcy = None

keyb = Controller()

capture = cv2.VideoCapture(0)

mediapipeHands = mediapipe.solutions.hands

myHands = mediapipeHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

drawHands = mediapipe.solutions.drawing_utils

while True:
    found, img = capture.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    foundHands = myHands.process(imgRGB)

    if foundHands.multi_hand_landmarks:
        actionDone = False
        for hands in foundHands.multi_hand_landmarks:
            for id, location in enumerate(hands.landmark):
                h, w, c = img.shape
                cx = int(location.x * w)
                cy = int(location.y * h)

                if pollCounter == 200: #compares user hand position right now vs 200 loops ago. gives time for the user to move
                    pollCounter = 0
                    if oldcx and oldcy and not actionDone:
                        #Swipe up
                        if abs(oldcx - cx) < 100 and (oldcy - cy) > 100:
                            keyb.press(Key.space)
                            keyb.release(Key.space)
                            actionDone = True

                        #Swipe left --> comp goes right
                        elif (oldcx - cx) > 150 and abs(oldcy - cy) < 100:
                            keyb.press(Key.left)
                            keyb.release(Key.left)
                            actionDone = True

                        #Swipe right --> comp goes left
                        elif (oldcx - cx) < -150 and abs(oldcy - cy) < 100:
                            keyb.press(Key.right)
                            keyb.release(Key.right)
                            actionDone = True
                    
                    oldcx = cx
                    oldcy = cy
                pollCounter += 1

            drawHands.draw_landmarks(img, hands, mediapipeHands.HAND_CONNECTIONS) #draw landmarks on hand

            cv2.putText(img,str("X: "+str(cx)+" Y: "+str(cy)), (10,70), cv2.QT_FONT_NORMAL, 1, (0,0,255), 1) #displays x and y of hand
    
    cv2.imshow("MangaReader", img) #show camera

    if cv2.waitKey(1) & 0xFF == ord('q'): #quit program if user presses q
        break