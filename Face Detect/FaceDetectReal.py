import cv2
import numpy as np 
import os

#  Our KNN algorithm
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

#Code Related to Cv2
skip = 0
face_data = []			  #Used to store the training data of each label
labels = []
dataset_path = './data/'  #Path where we put our training data
class_id = 0			  #Used to create id for earch classes of training data
names = {}				  #Used to map id to the names of classes
cap = cv2.VideoCapture(0)
cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


# Prepration of data

for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)
		# how to create label? ->
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset =  np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape(-1,1)
trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Display Output

while True:
	ret,frame = cap.read()
	if ret == False:
		continue
	faces = cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key = lambda f:f[2]*f[3])

	for (x,y,w,h) in faces[-1:]:
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		#resize the image into 100x100 image
		face_section = cv2.resize(face_section,(100,100))
		#output prediction from KNN:
		out = knn(trainset,face_section.flatten())
		# Obtain label from dictionary
		name = names[int(out)]
		cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+w),(255,255,122),2)

	cv2.imshow("Facial Recognition",frame)
	key = cv2.waitKey(1) 

	if key == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()