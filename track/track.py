
'''
    © Copyright (C) 2023
    Collaborateurs : Amine YOUCEF KHODJA, Koceila KEMICHE, Hoang Son NGUYEN.*

'''


import cv2 as cv

v = cv.VideoCapture('video_boite_entiere-test.mp4')
bg = cv.imread("bg.png")
bg = cv.cvtColor(bg,cv.COLOR_BGR2GRAY)
bg = cv.GaussianBlur(bg, (21,21),0)


while True: # lire frame par frame
	ok, frame = v.read()

	gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY) # couleur -> gray
	gray = cv.GaussianBlur(gray, (21,21),0) # enlever noise kernel (21,21)

	diff = cv.absdiff(bg,gray) # comparer avec le background
	thresh = cv.threshold(diff,35,255,cv.THRESH_BINARY)[1] # si sup a 30 -> 255, sinon 0
	thresh = cv.dilate(thresh,None,iterations = 2) # rendre plus epaise
	cnts, res = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # trouver contours dans thresh

	for c in cnts: # tracer box
		(x,y,w,h) = cv.boundingRect(c) # coord
		title = "fourmis "+str(idd)
		cv.putText(frame,title,(x,y),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
		cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)


	cv.imshow("frame",frame)
	# cv.imshow("thresh",thresh)
	key = cv.waitKey(1)
	if key == ord('q'):
		break

v.release()
cv.destroyWindows()


 
