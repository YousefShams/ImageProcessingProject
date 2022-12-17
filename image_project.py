from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import cv2
import numpy as np


#GLOBAL Variables

filters = [np.full([3,3], 1/9), np.full([5,5], 1/25) , np.full([7,7], 1/49)]

currentFilter = filters[0]

#our default (rgb) colour
bg="#100f1f"

#default font
myFont=("Calibri", 30, "bold")

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
  resizedImage = cv2.resize(imageWithPadding, (0,0), fx=1/4 , fy=1/4)


  nrows , ncols , nchannels  = resizedImage.shape


  # the output of this function is a list of blue , green and red matrices of the input image
  bgrInChannels = cv2.split(resizedImage) 


  # initializing r,g,b matrices with zeros, they will be filled with values later
  bgrOutChannels = [ np.zeros([nrows-2, ncols-2]) for i in range(nchannels) ]


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

    inputImagePath = filedialog.askopenfilename(title= "Open Image")
    inputImage = ImageTk.PhotoImage(Image.open(inputImagePath).resize(imageDims))
    canvas1.itemconfig(inputImageBox, image=inputImage)
    
    outputImage =  ImageTk.PhotoImage(file="./placeholder.png")
    canvas2.itemconfig(outputImageBox, image=outputImage)
    
    
def ProcessImage():
    
    if(inputImagePath.find("placeholder") == -1) :
      global outputImage
      outputImageCV = ApplyFilter(inputImagePath, currentFilter )
      
      outputImageName =  "output_"+ inputImagePath.split("/")[-1].split(".")[0]
      outputImagePath = outputPath + outputImageName + ".png"
      
      
      
      outputImage = ImageTk.PhotoImage(Image.fromarray(np.uint8(outputImageCV[:,:,::-1].tolist())).resize(imageDims))
      canvas2.itemconfig(outputImageBox, image=outputImage)


def ShowHistogram():

  global outputImage

  # Reading the image
  inputImage = cv2.imread(inputImagePath, 1)
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

  outputImage = ImageTk.PhotoImage(Image.open(outputImagePath).resize(imageDims))
  canvas2.itemconfig(outputImageBox, image=outputImage)
  

def SelectFilter(value):
  global currentFilter
  currentFilter = filters[value]


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
submenu = Menu(window) # Another submenu that appears after clicking pervious submenu

menuBar.add_cascade(label="Menu", menu=menu1)

menu1.add_cascade(label="Open Image", command= OpenImage)
menu1.add_cascade(label="Filters", menu=submenu)

submenu.add_radiobutton(label="Gaussian Filter", command= lambda: SelectFilter(0) )
submenu.add_radiobutton(label="Laplacian Filter" , command= lambda: SelectFilter(1))
submenu.add_radiobutton(label="Noise Removal Filter", command= lambda: SelectFilter(2))



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
Button(f4, text="Process Image", font=myFont, borderwidth = 0, command=lambda:ProcessImage()).pack(side=LEFT,padx=10)
Button(f4, text="Show Histogram", font=myFont, borderwidth = 0, command= ShowHistogram).pack(side=LEFT,padx=10)
Button(f4, text="Quit", font=myFont, borderwidth = 0, bg="red", command=lambda : Quit()).pack(side=BOTTOM,padx=10)




f1.pack()
f2.pack(padx=50) 
f3.pack(padx= 50)
f4.pack(pady= 50)


window.bind("<Escape>", smallScreen)
window.bind("<f>", fullScreen)


window.mainloop()

