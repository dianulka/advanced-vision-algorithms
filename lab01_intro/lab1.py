import cv2
I = cv2.imread('mandril.jpg')
cv2.imshow("I", I)
cv2.waitKey(0)
cv2.destroyAllWindows()

IG = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
IHSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)