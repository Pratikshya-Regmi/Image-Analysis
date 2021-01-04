# USAGE
# tkinter_test.py

## import the necessary packages
from Tkinter import *
import numpy as np
from PIL import Image
from PIL import ImageTk
import tkFileDialog
import cv2
import tkMessageBox
from matplotlib import pyplot as plt

def canny():
	# grab a reference to the image panels
	global panelA, panelB

	# open a file chooser dialog and allow the user to select an input
	# image
	path = tkFileDialog.askopenfilename()

	# ensure a file path was selected
	if len(path) > 0:
		## load the image from disk, convert it to grayscale, and detect
		## edges in it
		image = cv2.imread(path)

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edged=cv2.Canny(gray,100,100)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


		## convert the images to PIL format...
		image = Image.fromarray(image)
		edged = Image.fromarray(edged)

		## ...and then to ImageTk format
		image = ImageTk.PhotoImage(image)
		edged = ImageTk.PhotoImage(edged)


		# if the panels are None, initialize them
		if panelA is None or panelB is None:
			# the first panel will store our original image
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)

			# while the second panel will store the edge map
			panelB = Label(image=edged)
			panelB.image = edged
			panelB.pack(side="right", padx=10, pady=10)

		## otherwise, update the image panels
		else:
			# update the pannels
			panelA.configure(image=image)
			panelB.configure(image=edged)
			panelA.image = image
			panelB.image = edged




		## OpenCV represents images in BGR order; however PIL represents
		## images in RGB order, so we need to swap the channels


def count_object():
    global panelA, panelB
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        edged = cv2.Canny(gray, 100, 100)
        cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Input Image", image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(221),plt.imshow(image,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(edged,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imgray, (5, 5), 0)
        ret,thresh = cv2.threshold(blur,127,255,0,cv2.THRESH_BINARY)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        kernel = np.ones((10,10),np.uint8)
        dilation = cv2.dilate(edged,kernel,iterations = 1)
        #kernele=np.ones((5,5),np.uint8)
        #erosion = cv2.erode(dilation,kernele,iterations = 1)
        img = cv2.drawContours(dilation, contours, -1, (0,255,0), 3)
        plt.subplot(223),plt.imshow(dilation,cmap = 'gray')
        plt.title('Dilated Image'), plt.xticks([]), plt.yticks([])
        (_, contours, _) = cv2.findContours(img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print("Found %d objects." % len(contours))
        w=tkMessageBox.showinfo("Object Count","Found %d objects." % len(contours))

        c=cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
        cv2.waitKey(0)
        plt.subplot(224),plt.imshow(c,cmap = 'gray')
        plt.title('Contour Image'), plt.xticks([]), plt.yticks([])
        plt.show()


def laplacian():
    global panelA, panelB
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        edged = cv2.Laplacian(gray,cv2.CV_64F,ksize=3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)



        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
            # otherwise, update the image panels
        else:			# update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged

def sobel():
    global panelA, panelB
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        edged = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        image = Image.fromarray(image)
        edged = Image.fromarray(edged)
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)



        if panelA is None or panelB is None:
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)
            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)
            # otherwise, update the image panels
        else:			# update the pannels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged


def face_detect():
    global panelA, panelB
    path = tkFileDialog.askopenfilename()
    if len(path) > 0:
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h, x:x+w]
        roi_color=image[y:y+h,x:x+w]
    cv2.imshow('image',image)





# initialize the window toolkit along with the two image panels
root = Tk()
root.minsize(200,200)
#Label(root,text='Image Processing applications').pack()
panelA = None
panelB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI


def Button_packer():
    Cannybutton = Button(root,fg='red' ,relief='groove',text="Canny Edge Detection", command=canny)

    Cannybutton.pack(side="bottom", fill="both", expand="yes", padx="20", pady="10")


    Sobelbutton=Button(root, fg='red',relief='groove', text="Sobel Edge Detection", command=sobel)
    Sobelbutton.pack(side="bottom", fill="both", expand="yes", padx="20", pady="10")

    Laplacianbutton=Button(root, fg='red',relief='groove', text="Laplacian Edge Detection", command=laplacian)
    Laplacianbutton.pack(side="bottom", fill="both", expand="yes", padx="20", pady="10")



CountButton= Button(root, text="Count Objects", command=count_object)
CountButton.pack(side="top", fill="both", expand="yes", padx="400", pady="10")

FaceDetectbutton= Button(root, text="Detect Faces", command=face_detect)
FaceDetectbutton.pack(side="top", fill="both", expand="yes", padx="400", pady="10")

edgedetectbutton = Button(root, text="Detect edges ", command=Button_packer)
edgedetectbutton.pack(side="top", fill="both", expand="yes", padx="400", pady="10")








Label(root,text='           3.Detect Edges to extract an image showing edges of objects in the input image using respective algorithms:  a.Laplacian    b.Sobel     c.Canny').pack(side="bottom",anchor=W)
Label(root,text='           2.Detect Faces to recognize face in a group or single photo').pack(side="bottom",anchor=W)
Label(root,text='Select:1.Count Objects to count the number of inanimate objects in your image').pack(side="bottom",anchor=W)
my_image=cv2.imread('040E68C1000003E8-0-image-m-58_1502711246235.jpg')
#canvas = Canvas(root, width=500, height=70).pack()
root.configure(background='#a1dbcd')


# kick off the GUI
root.mainloop()
