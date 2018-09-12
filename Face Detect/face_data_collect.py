import cv2
import numpy as np
file_name = input("Enter the name of person")
cap = cv2.VideoCapture(0)
#face Detection haarcascade
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
face_data = []
dataset_path = './data/'
while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	if ret == False:
		continue
	faces = cascade.detectMultiScale(frame
		,1.3,5)
	faces = sorted(faces,key = lambda f:f[2]*f[3])
	face_section = frame
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,200,100),2)
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		#resize the image into 100x100 image
		face_section = cv2.resize(face_section,(100,100))
		skip +=1
		# Add every 10th image
		if (skip%20) == 0:
			face_data.append(face_section)
			# print(face_data[-1])
			print(len(face_data))
	cv2.imshow("Face",frame)
	# cv2.imshow("Section",face_section)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'):
		break
#store the captured faces as numpy array

face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0],-1)
# # Saving file into system
np.save(dataset_path+file_name+'.npy',face_data)

cap.release()
cv2.destroyAllWindows()