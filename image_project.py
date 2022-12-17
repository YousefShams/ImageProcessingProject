from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


#GLOBAL Variables
laplacianFilter = np.array([
  [1,1,1],
  [1,-8,1],
  [1,1,1],
])

gaussiannFilter = np.array([
  [1,2,1],
  [2,4,2],
  [1,2,1],
]) / 16

avergaeFilter = np.array([
  [1,1,1],
  [1,1,1],
  [1,1,1],
  ]) /9

filters = [gaussiannFilter, laplacianFilter , avergaeFilter]

currentFilter = filters[0]
histType = 1

#our default (rgb) colour
bg="#100f1f"

#default fonts
defaultFont=("Calibri", 30, "bold")
buttonFont = ("Calibri", 20, "bold")
menuFont = ("Calibri", 10, "bold")

imageDims = [400,300]

inputImagePath = "./placeholder.png"
outputImagePath = "./placeholder.png"
outputPath = "./outputs/"







#LOGIC Part

def fullScreen (event) :
    window.attributes("-fullscreen", True)
    
    
def smallScreen (event) :
    window.attributes("-fullscreen", False)


def ApplyFilter(inputImagePath, filter):

  #Reading the image
  inputImage = cv2.imread(inputImagePath, 1)
      
    
  # giving the left,bottom, top and right of an image 1 padding (same padding)
  imageWithPadding = cv2.copyMakeBorder(inputImage,1,1,1,1,cv2.BORDER_REPLICATE)

  # resizing the image down before processing, so the processing is faster
  resizedImage = cv2.resize(imageWithPadding, imageDims)


  nrows , ncols , nchannels  = resizedImage.shape


  # the output of this function is a list of blue , green and red matrices of the input image
  bgrInChannels = cv2.split(resizedImage) 


  # initializing r,g,b matrices with zeros, they will be filled with values later
  bgrOutChannels = [ np.zeros([nrows-2, ncols-2], dtype=np.uint8) for i in range(nchannels) ]


  for channelIndex in range(nchannels) :
    for r in range(1,nrows-1):  # r for row
      for c in range(1,ncols -1): # c for column
      
        sum = 0
        for r_filter in range(3): 
          for c_filter in range(3):
            sum += bgrInChannels[channelIndex][r + r_filter -1] [c + c_filter -1] * filter[r_filter][c_filter]
          
        bgrOutChannels[channelIndex] [r-1] [c-1] = sum
  
  # merging the blue , green and red matrices to form the color image
  output_image = cv2.merge(bgrOutChannels)
  # resizing the image up again (so it has its normal size) and then returning it
  return output_image


def OpenImage():
    
    global inputImage
    global inputImagePath
    global outputImage
    
    previnputImagePath = inputImagePath
    
    #function that opens a file dialog and returns the seleected file path
    inputImagePath = filedialog.askopenfilename(title= "Open Image")    
    
    #if the returned path is empty, take the value of the previous path
    if(inputImagePath==""): inputImagePath = previnputImagePath
    
    #updating the user interface input image with the the selected image from the input path
    inputImage = ImageTk.PhotoImage(Image.open(inputImagePath).resize(imageDims))
    canvas1.itemconfig(inputImageBox, image=inputImage)
    
    outputImage =  ImageTk.PhotoImage(file="./placeholder.png")
    canvas2.itemconfig(outputImageBox, image=outputImage)
    
    
def ProcessImage(outputImageCV):
  
  global outputImage
  
  if len(outputImageCV.shape) == 3:
    outputImage = ImageTk.PhotoImage(  Image.fromarray(outputImageCV[:,:,::-1]).resize(imageDims)  ) 
    
  else:
    outputImage = ImageTk.PhotoImage(  Image.fromarray(outputImageCV).resize(imageDims)   )
    
  canvas2.itemconfig(outputImageBox, image=outputImage)    
  

def ConvertToGrayScale(InputImagePath):
  
  coloured_img = cv2.resize(cv2.imread(InputImagePath) , imageDims)
  gray_img = np.zeros(coloured_img.shape[0:2])

  (row,col) = coloured_img.shape[0:2]
  
  for i in range(row):
    for j in range(col):
      gray_img[i,j] = sum(coloured_img[i,j]) * 0.33
  return gray_img


def ShowHistogram(type):
  if type==1:
    return ShowHistogram1(inputImagePath)
  else:
    return ShowHistogram2(inputImagePath)
  

def ShowHistogram1(inputImagePath):
  Histogram = HistogramComputation(inputImagePath)
  return PlotHistogram(Histogram)
  

def ShowHistogram2(inputImagePath):
  inputImage = cv2.resize(cv2.imread(inputImagePath, 1), imageDims)
  # info about the image (height, width, number of channels)
  nrows , ncols , nchannels = inputImage.shape
  # initial values of the percentage of blue, green and red colors in the image
  colorsRatios = [0,0,0]

  numOfPixels = nrows * ncols  # nrows is the same as the width, ncolss is the same as the height

  for colorNum in range(3):
    sumOfColor = 0

    for row in range(nrows):
      for col in range(ncols):
        sumOfColor += inputImage[row][col][colorNum]

    colorsRatios[colorNum] = sumOfColor / numOfPixels

  x = ["Blue", "Green", "Red"]
  y = colorsRatios.copy()

  plt.bar(x[0],y[0], color="b")
  plt.bar(x[1],y[1], color="g")
  plt.bar(x[2],y[2], color="r")

  outputImageName =  "output_"+ inputImagePath.split("/")[-1].split(".")[0]
  outputImagePath = outputPath + outputImageName + ".png"
  plt.savefig(outputImagePath, bbox_inches='tight')
  
  return cv2.imread(outputImagePath)

  
def HistogramComputation(inputImagePath):                                               #basically its a looping function over the converted image pixels with 3 channels
	
	Image = cv2.imread(inputImagePath)
	ImageHeight, ImageWidth, ImageChannels = Image.shape
	
	Histogram = np.zeros([256, ImageChannels], np.int32)                          # creating empty array for each channel filled with zeros
	
	for h in range(0, ImageHeight):
		for w in range(0, ImageWidth):
			for c in range(0, ImageChannels):
				colorValue = Image[h,w,c]
				Histogram[ colorValue , c] +=1
	
	return Histogram


def PlotHistogram(Histogram):
  
    global outputImagePath
    plt.figure()
    
    plt.title("Color Image Histogram")
    plt.xlabel("Intensity Level")
    plt.ylabel("Intensity Frequency")
    plt.xlim([0, 256])                                                            # specifies automatic or manual limit selection.
    
    plt.plot(Histogram[:,0],'b')                                 # This is to Plot Blue Channel with Blue Color
    plt.plot(Histogram[:,1],'g')                                 # This is to Plot Green Channel with Green Color
    plt.plot(Histogram[:,2],'r')                                 # This is to Plot Red Channel with Red Color
    
    outputImageName =  "output_"+ inputImagePath.split("/")[-1].split(".")[0]
    outputImagePath = outputPath + outputImageName + ".png"
    plt.savefig(outputImagePath, bbox_inches='tight')
    
    return cv2.imread(outputImagePath)


def ExtractColor(inputImagePath,colorNum):
  resize_image = cv2.resize(cv2.imread(inputImagePath), imageDims)
  b,g,r = resize_image[:,:,0] , resize_image[:,:,1] , resize_image[:,:,2]
  
  blank = np.zeros(resize_image.shape[0:2], dtype= np.uint8)
  
  if(colorNum == 0):
    return cv2.merge([b,blank,blank])
  
  elif(colorNum == 1):
    return cv2.merge([blank, g ,blank])
  
  else:
    return cv2.merge([blank, blank, r])
    

def SelectFilter(value):
  global currentFilter
  currentFilter = filters[value]


def SelectHistogram(value):
  global histType
  histType = value


def Quit():
  window.destroy()








#UI Part


window = Tk()
window.attributes("-fullscreen", False)
window.geometry("1200x900")
window["background"] = bg
window.title("Image Editor")


# Creating Menu Elements
menuBar = Menu(window) # Menu Top Bar
menu1 = Menu(window) # Submenu that appears after clicking menu top bar
menu2 = Menu(window)
submenu = Menu(window) # Another submenu that appears after clicking pervious submenu
submenu2 = Menu(window)

menuBar.add_cascade(label="Open", font=menuFont, command= OpenImage)
menuBar.add_cascade(label="Modes", menu=menu1)
menuBar.add_cascade(label="Exctract Color", menu=menu2)

menu1.add_cascade(label="Filter Types", menu=submenu)
menu1.add_cascade(label="Histogram Types", menu=submenu2)

menu2.add_cascade(label="Blue", command= lambda : ProcessImage( ExtractColor(inputImagePath, 0) ))
menu2.add_cascade(label="Green", command= lambda : ProcessImage( ExtractColor(inputImagePath, 1) ))
menu2.add_cascade(label="Red", command= lambda : ProcessImage( ExtractColor(inputImagePath, 2) ))


submenu.add_radiobutton(label="Gaussian Filter", command= lambda: SelectFilter(0) )
submenu.add_radiobutton(label="Laplacian Filter" , command= lambda: SelectFilter(1))
submenu.add_radiobutton(label="Noise Removal Filter", command= lambda: SelectFilter(2))

submenu2.add_radiobutton(label="Bar", command= lambda: SelectHistogram(1) )
submenu2.add_radiobutton(label="Plot" , command= lambda: SelectHistogram(2))



window.config(menu=menuBar)


#Creating Some Frames
#you can imagine the frame as a box that contains some elements like : (text,images and so on...)

f1 = Frame(background=bg)
f2 = Frame(background=bg)
f3 = Frame(background=bg)
f4 = Frame(background=bg)


#Creating Elements for Frame1
Label(f1 ,text="Image Editor", bg=bg,  fg="white", font=("Cooper Black", 40, "bold"), pady=50).pack() #Text Widget


#Creating Elements for Frame2
Label(f2 ,text="Input Image", bg=bg,  fg="white", font=("Cooper Black", 25, "bold")).pack(side=LEFT)
Label(f2 ,text="", bg=bg).pack(side=LEFT, padx=190)
Label(f2 ,text="Output Image", bg=bg,  fg="white", font=("Cooper Black", 25, "bold")).pack(side=LEFT)

#Creating Elements for Frame3
inputImage = ImageTk.PhotoImage(file=inputImagePath)
outputImage = ImageTk.PhotoImage(file=outputImagePath)

canvas1 = Canvas(f3,width=400, height=300)
canvas1.pack(side=LEFT,)

Label(f3 ,text="", bg=bg).pack(side=LEFT, padx=100)

canvas2 = Canvas(f3, width=400, height=300)
canvas2.pack(side=LEFT,pady = 50)

inputImageBox = canvas1.create_image(0,0,anchor="nw", image=inputImage)
outputImageBox = canvas2.create_image(0,0,anchor="nw", image=outputImage)



#Creating Elements for Frame4
Button(f4, text="Apply Filter", font=buttonFont, borderwidth = 0, command=lambda:ProcessImage(ApplyFilter(inputImagePath, currentFilter))).pack(side=LEFT,padx=10)
Button(f4, text="Display Histogram", font=buttonFont, borderwidth = 0, command= lambda : ProcessImage(ShowHistogram(histType))).pack(side=LEFT,padx=10)
Button(f4, text="Convert to Grayscale", font=buttonFont, borderwidth = 0, command= lambda: ProcessImage(ConvertToGrayScale(inputImagePath))).pack(side=LEFT,padx=10)
Button(f4, text="Quit", font=buttonFont, borderwidth = 0, bg="red", command=lambda : Quit()).pack(side=BOTTOM,padx=10)




f1.pack()
f2.pack(padx=50) 
f3.pack(padx= 50)
f4.pack(pady= 50)



window.bind("<Escape>", smallScreen)
window.bind("<f>", fullScreen)


window.mainloop()

