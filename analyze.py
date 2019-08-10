import dlib
import imutils.face_utils as utils
import scipy.spatial.distance as distance
import cv2
import time

DEFAULT_SHAPE_PRED_PATH = "shape_predictor_68_face_landmarks.dat"
VIDEO_SOURCE = 0
EYE_ASPECT_RATIO_THRESHOLD = 0.3

class Analyze:
	#Drowsiness analyzer class
	def __init__(self,shapePredictorPath=DEFAULT_SHAPE_PRED_PATH,videoSource = VIDEO_SOURCE):
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(shapePredictorPath)
		self.leftEyeLandMark = utils.FACIAL_LANDMARKS_IDXS['left_eye']
		self.rightEyeLandMark = utils.FACIAL_LANDMARKS_IDXS['right_eye']
		self.videoSource = videoSource
		#self.videoCapture = cv2.VideoCapture(videoSource)

	def get_ear_ratio_from_eyes(self,eyeLandMarks):
		leftRegion = distance.euclidean(eyeLandMarks[1], eyeLandMarks[5])
		rightRegion = distance.euclidean(eyeLandMarks[2], eyeLandMarks[4])
		middleDivider = distance.euclidean(eyeLandMarks[0], eyeLandMarks[3])
		return (leftRegion+rightRegion)/(2*middleDivider)

	def get_frame():
		#get frame from the source set in object constructor.
		ret, frame = self.videoCapture.read()
		frame = cv2.flip(frame,1)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		return gray_frame

	def get_ear_ratio_from_face(self,grayFrame,face):
		#Given a face,returns the ear ratio
		shape = self.predictor(grayFrame, face)
		shape = utils.shape_to_np(shape)

		leftEye = shape[self.leftEyeLandMark[0]:self.leftEyeLandMark[1]]
		rightEye = shape[self.rightEyeLandMark[0]:self.rightEyeLandMark[1]]
		return (self.get_ear_ratio_from_eyes(leftEye) + self.get_ear_ratio_from_eyes(rightEye))/2

	def get_ear_ratio_from_frame(self,frame):
		faces = self.detector(frame, 1)
		ear_ratio = []
		for face in faces:
			ratio = self.get_ear_ratio_from_face(frame,face)
			ear_ratio.append(ratio)
		return ear_ratio

ana = Analyze()
videoCapture = cv2.VideoCapture(0)

while(True):
	ret, frame = videoCapture.read()
	frame = cv2.flip(frame,1)
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	ratios = ana.get_ear_ratio_from_frame(gray_frame)
	if (sum(ratios) < EYE_ASPECT_RATIO_THRESHOLD):
		print("Sleepy Head")
	time.sleep(2)



	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break

