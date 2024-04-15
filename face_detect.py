import PySimpleGUI as sg
import cv2

layout = [
	[sg.Image(key = '-IMAGE-')],
	[sg.Text('People in picture: 0', key = '-TEXT-', expand_x = True, justification = 'c')]		# c stands for center
]

window = sg.Window('Face Detector', layout)

# get video
video = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
	event, values = window.read(timeout = 20)
	if event == sg.WIN_CLOSED:
		break

	_, frame = video.read()		# the '_' is boolean for success of read(), we only concerned with the video frame
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)	# cvtColor: convert the color of frame to gray

	# Reason for conversion to grayscale:
		# Haar cascades used in OpenCV are designed to work with grayscale images

	faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor = 1.3, 		# scales down the image repeatedly by a factor of 1.3, This multi-scale approach ensures that faces of various sizes can be detected in the image.
		minNeighbors = 7,
		minSize =(30,30))

	# draw the rectangles
	for (x, y, w, h) in faces:
		cv2.rectangle(frame,(x,y),(x + w, y + h),(0,255,0),2)

	# update the image
	imgbytes = cv2.imencode('.png',frame)[1].tobytes()
	window['-IMAGE-'].update(data = imgbytes)

	# update the text
	window['-TEXT-'].update(f'People in picture: {len(faces)}')

window.close()

