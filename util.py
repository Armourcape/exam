import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#def util(img_path, vid_path, mask_path):
def readimgvid(img_path, vid_path):    
    # reading img and videos
    img = cv.imread(img_path)
    cv.imshow('img', img)
    cv.waitKey(0)
    capture = cv.VideoCapture(vid_path)
    while True:
        isTrue, frame = capture.read()
        if isTrue:    
            cv.imshow('Video', frame)
            if cv.waitKey(20) & 0xFF==ord('d'):
                break            
        else:
            break
    capture.release()
    cv.destroyAllWindows()

    # image transformations

def trans_translate(img_path):
    import cv2 as cv
    import numpy as np
    img=cv.imread(img_path,0)
    rows,cols=img.shape
    M=np.float32([[1,0,100],[0,1,50]])
    dst=cv.warpAffine(img,M,(cols,rows))
    cv.imshow('Image',dst)
    cv.waitKey(0)
    cv.destroyAllWindows
    
def trans_reflection_xaxis(img_path):
    import numpy as np
    import cv2 as cv
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    #M = np.float32([[1,  0, 0],[0, -1, rows],[0,  0, 1]]) #vertical
    M = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]]) #horizontal
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    cv.imshow('img', reflected_img)
    cv.imwrite('reflection_out.jpg', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows()  
    
def trans_reflection_yaxis(img_path):
    import numpy as np
    import cv2 as cv
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1,  0, 0],[0, -1, rows],[0,  0, 1]]) #vertical
    #M = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]]) #horizontal
    reflected_img = cv.warpPerspective(img, M,(int(cols),int(rows)))
    cv.imshow('img', reflected_img)
    cv.imwrite('reflection_out.jpg', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows() 

def trans_rotate(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1,  0, 0], [0, -1, rows], [0,  0, 1]])
    img_rotation = cv.warpAffine(img,cv.getRotationMatrix2D((cols/2, rows/2),30, 0.6),(cols, rows))
    cv.imshow('img', img_rotation)
    cv.imwrite('rotation_out.jpg', img_rotation)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def trans_shrink(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    img_shrinked=cv.resize(img, (250, 200),interpolation=cv.INTER_AREA)
    cv.imshow('img', img_shrinked)
    img_enlarged = cv.resize(img_shrinked, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def trans_enlarge(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    img_shrinked=cv.resize(img, (250, 200),interpolation=cv.INTER_AREA)
    img_enlarged = cv.resize(img_shrinked, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    img_enlarged = cv.resize(img_enlarged, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    img_enlarged = cv.resize(img_enlarged, None,fx=1.5, fy=1.5,interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()

def trans_crop(img_path):
    img=cv.imread(img_path)
    cv.imshow('img',img)
    cropped=img[50:200,200:400]
    cv.imshow('Cropped',cropped)
    cv.waitKey(0)

def trans_x_shearing(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('img', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def trans_y_shearing(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1,   0, 0], [0.5, 1, 0], [0,   0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('sheared_y-axis_out.jpg', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows() 
    
def trans_contours(img_path):
    img=cv.imread(img_path)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edge=cv.Canny(gray,30,300)
    contours,hierarchy=cv.findContours(edge,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    cv.imshow('Canny edges after contouring',edge)
    print('Number of contours found=',str(len(contours)))
    cv.drawContours(img,contours,-1,(0,255,0),3) #-1 signifies drawing all contours
    cv.imshow('contours',img)
    cv.waitKey(0)
    cv.destroyAllWindows

def shapes():
    # drawing shapes

    blank = np.zeros((500,500,3), dtype='uint8')
    cv.imshow('Blank', blank)
    blank[200:300, 300:400] = 0,0,255
    cv.imshow('Green', blank)
    cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,255,0), thickness=-1)
    cv.imshow('Rectangle', blank)
    cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (0,0,255), thickness=-1)
    cv.imshow('Circle', blank)
    cv.line(blank, (100,250), (300,400), (255,255,255), thickness=3)
    
def text():
    blank = np.zeros((500,500,3), dtype='uint8')
    cv.imshow('Line', blank)
    cv.putText(blank, 'Hello, my name is Jason!!!', (0,225), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,0), 2)
    cv.waitKey(0)
    # putting text

    cv.imshow('Text', blank)
    cv.waitKey(0)

def resize(img_path, vid_path):

    # Resizing and Rescaling Frames

    img = cv.imread(img_path)
    cv.imshow('img', img)

    def rescaleFrame(frame, scale=0.75):
        width = int(frame.shape[1]*scale)
        height = int(frame.shape[0]*scale)

        dimensions = (width,height)

        return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)

    resized_image = rescaleFrame(img,scale=0.2)
    cv.imshow('Image resized', resized_image)


    #Reading Videos
    capture = cv.VideoCapture(vid_path)

    while True:
        isTrue, frame = capture.read()

        frame_resized = rescaleFrame(frame,scale=0.2)

        cv.imshow('video', frame)
        cv.imshow('video Resized', frame_resized)


        if cv.waitKey(20) & 0xFF==ord('d'):
            break


    capture.release()
    cv.destroyAllWindows

    # Essential functions in opencv
def ess_gray(img_path):
        # Converting to grayscale
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)
    return gray
def ess_blur(img_path):
        # Blur
    img = cv.imread(img_path) 
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
    cv.imshow('Blur', blur)
def ess_edges(img_path):
        # Edge Cascade
    img = cv.imread(img_path) 
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)
    cv.imshow('Canny Edges', canny)
def ess_dilate(img_path):
        # Dilating the image
    img = cv.imread(img_path) 
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)
    dilated = cv.dilate(canny, (7,7), iterations=3)
    cv.imshow('Dilated', dilated)
def ess_erode(img_path):
        # Eroding
    img = cv.imread(img_path) 
    blur = cv.GaussianBlur(img, (7,7), cv.BORDER_DEFAULT)
    canny = cv.Canny(blur, 125, 175)
    dilated = cv.dilate(canny, (7,7), iterations=3)
    eroded = cv.erode(dilated, (7,7), iterations=3)
    cv.imshow('Eroded', eroded)
def ess_resize(img_path):
        # Resize
    img = cv.imread(img_path)
    resized = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
    cv.imshow('Resized', resized)
def ess_crop(img_path):
        # Cropping
    img = cv.imread(img_path)
    cropped = img[50:200, 200:400]
    cv.imshow('Cropped', cropped)
    cv.waitKey(0)
def translation(img_path):
    # Image Translation
    
    img = cv.imread(img_path)
    def translate(img, x, y):
        transMat = np.float32([[1,0,x],[0,1,y]])
        dimensions = (img.shape[1], img.shape[0])
        return cv.warpAffine(img, transMat, dimensions)

    # -x --> Left
    # -y --> Up
    # x --> Right
    # y --> Down

    translated = translate(img, -100, 100)
    cv.imshow('Translated', translated)
    cv.waitKey(0)
def reflection(img_path):
    # Reflection

    img = cv.imread(img_path)
    rows, cols, channels = img.shape
    M = np.float32([[1,  0, 0],
                    [0, -1, rows],
                    [0,  0, 1]])
    reflected_img = cv.warpPerspective(img, M,
                                    (int(cols),
                                        int(rows)))
    cv.imshow('img', reflected_img)
    cv.imwrite('reflection_out.jpg', reflected_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def rotation(img_path):
    # Image Rotation

    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1,  0, 0], [0, -1, rows], [0,  0, 1]])
    img_rotation = cv.warpAffine(img,
                                cv.getRotationMatrix2D((cols/2, rows/2),
                                                        30, 0.6),
                                (cols, rows))
    cv.imshow('img', img_rotation)
    cv.imwrite('rotation_out.jpg', img_rotation)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Image Scaling
def scaling(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    img_shrinked = cv.resize(img, (250, 200),
                            interpolation=cv.INTER_AREA)
    cv.imshow('img', img_shrinked)
    img_enlarged = cv.resize(img_shrinked, None,
                            fx=1.5, fy=1.5,
                            interpolation=cv.INTER_CUBIC)
    cv.imshow('img', img_enlarged)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    # Image Cropping
def crop(img_path):
    img = cv.imread(img_path, 0)
    cropped_img = img[100:300, 100:300]
    cv.imwrite('cropped_out.jpg', cropped_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Image Shearing in X-Axis
def shear_x(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('img', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Image Shearing in Y-Axis
def shear_y(img_path):
    img = cv.imread(img_path, 0)
    rows, cols = img.shape
    M = np.float32([[1,   0, 0], [0.5, 1, 0], [0,   0, 1]])
    sheared_img = cv.warpPerspective(img, M, (int(cols*1.5), int(rows*1.5)))
    cv.imshow('sheared_y-axis_out.jpg', sheared_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # contours
def contours(img_path):
    image = cv.imread(img_path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edged = cv.Canny(gray, 30, 200)
    cv.waitKey(0)
    contours, hierarchy = cv.findContours(edged, 
        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    cv.imshow('Canny Edges After Contouring', edged)
    cv.waitKey(0)
    
    print("Number of Contours found = " + str(len(contours)))
    
    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    
    cv.imshow('Contours', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # colour spaces
def colour_channels(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)

    B,G,R=cv.split(img)
    cv.imshow('original',img)
    cv.waitKey(0)
    cv.imshow('blue',B)
    cv.waitKey(0)
    cv.imshow('green',G)
    cv.waitKey(0)
    cv.imshow('red',R)
    cv.waitKey(0)

def colour_spaces(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)
        # BGR to Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Gray', gray)

        # BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('HSV', hsv)

        # BGR to L*a*b
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    cv.imshow('LAB', lab)

        # BGR to RGB
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    cv.imshow('RGB', rgb)

        # HSV to BGR
    lab_bgr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    cv.imshow('LAB --> BGR', lab_bgr)

    cv.waitKey(0)

    # blurring
def blur(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)

    # Averaging
    average = cv.blur(img, (3,3))
    cv.imshow('Average Blur', average)

    # Gaussian Blur
    gauss = cv.GaussianBlur(img, (3,3), 0)
    cv.imshow('Gaussian Blur', gauss)

    # Median Blur
    median = cv.medianBlur(img, 3)
    cv.imshow('Median Blur', median)

    # Bilateral
    bilateral = cv.bilateralFilter(img, 10, 35, 25)
    cv.imshow('Bilateral', bilateral)

    cv.waitKey(0)

    # bitwise
def bitwise(img_path, mask_path):
    img1 = cv.imread(img_path)  
    img2 = cv.imread(mask_path) 
    
    # cv2.bitwise_and is applied over the
    # image inputs with applied parameters 
    dest_and = cv.bitwise_and(img2, img1, mask = None)
    
    # the window showing output image
    # with the Bitwise AND operation
    # on the input images
    cv.imshow('Bitwise And', dest_and)
    
    # De-allocate any associated memory usage  
    if cv.waitKey(0) & 0xff == 27: 
        cv.destroyAllWindows()
# image inputs with applied parameters 
    dest_or = cv.bitwise_or(img2, img1, mask = None)
  
# the window showing output image
# with the Bitwise OR operation
# on the input images
    cv.imshow('Bitwise OR', dest_or)
    
    # De-allocate any associated memory usage  
    if cv.waitKey(0) & 0xff == 27: 
        cv.destroyAllWindows()

    dest_xor = cv.bitwise_xor(img1, img2, mask = None)
  
# the window showing output image
# with the Bitwise XOR operation
# on the input images
    cv.imshow('Bitwise XOR', dest_xor)
    
    # De-allocate any associated memory usage  
    if cv.waitKey(0) & 0xff == 27: 
        cv.destroyAllWindows()   
    
    dest_not1 = cv.bitwise_not(img1, mask = None)
    dest_not2 = cv.bitwise_not(img2, mask = None)
    
    # the windows showing output image
    # with the Bitwise NOT operation
    # on the 1st and 2nd input image
    cv.imshow('Bitwise NOT on image 1', dest_not1)
    cv.imshow('Bitwise NOT on image 2', dest_not2)
    
    # De-allocate any associated memory usage  
    if cv.waitKey(0) & 0xff == 27: 
        cv.destroyAllWindows()

    # masking
def mask(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)

    blank = np.zeros(img.shape[:2], dtype='uint8')
    cv.imshow('Blank Image', blank)

    circle = cv.circle(blank.copy(), (img.shape[1]//2 + 45,img.shape[0]//2), 100, 255, -1)

    rectangle = cv.rectangle(blank.copy(), (30,30), (370,370), 255, -1)

    weird_shape = cv.bitwise_and(circle,rectangle)
    cv.imshow('Weird Shape', weird_shape)

    masked = cv.bitwise_and(img,img,mask=weird_shape)
    cv.imshow('Weird Shaped Masked Image', masked)

    cv.waitKey(0)

    # histogram
def histogram(img_path):
    img = cv.imread(img_path)
    cv.imshow('img', img)

    blank = np.zeros(img.shape[:2], dtype='uint8')

    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)

    mask = cv.circle(blank, (img.shape[1]//2,img.shape[0]//2), 100, 255, -1)

    masked = cv.bitwise_and(img,img,mask=mask)
    cv.imshow('Mask', masked)

    # GRayscale histogram
    # gray_hist = cv.calcHist([gray], [0], mask, [256], [0,256] )

    # plt.figure()
    # plt.title('Grayscale Histogram')
    # plt.xlabel('Bins')
    # plt.ylabel('# of pixels')
    # plt.plot(gray_hist)
    # plt.xlim([0,256])
    # plt.show()

    # Colour Histogram

    plt.figure()
    plt.title('Colour Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    colors = ('b', 'g', 'r')
    for i,col in enumerate(colors):
        hist = cv.calcHist([img], [i], mask, [256], [0,256])
        plt.plot(hist, color=col)
        plt.xlim([0,256])

    plt.show()

    cv.waitKey(0)

def directory():
    print('readimgvid(img_path, vid_path)\ntrans_translate(img_path)\ntrans_reflection_xaxis(img_path)\ntrans_reflection_yaxis(img_path)\ntrans_rotate(img_path)\ntrans_shrink(img_path)\ntrans_enlarge(img_path)\ntrans_crop(img_path)\ntrans_x_shearing(img_path)\ntrans_y_shearing(img_path)\ntrans_contours(img_path)\nshapes()\ntext()\nresize(img_path, vid_path)\ness_gray(img_path)\ness_blur(img_path)\ness_edges(img_path)\ness_dilate(img_path)\ness_erode(img_path)\ness_resize(img_path)\ness_crop(img_path)\ntranslation(img_path)\nreflection(img_path)\nrotation(img_path)\nscaling(img_path)\ncrop(img_path)\nshear_x(img_path)\nshear_y(img_path)\ncontours(img_path)\ncolour_channels(img_path)\ncolour_spaces(img_path)\nblur(img_path)\nbitwise(img_path, mask_path)\nmask(img_path)\nhistogram(img_path)')
