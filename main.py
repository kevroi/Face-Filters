import sys
import numpy as np
import cv2
from scipy.interpolate import UnivariateSpline


# To run problem1(), in the command line enter:
# python main.py problem1 face1.jpg l 0.5
def problem1(src, mode, darkcoeff, blendcoeff):
    blendcoeff = float(blendcoeff)
    darkcoeff = float(darkcoeff)

    # Check if valid blending coefficient has been entered
    if (0 <= blendcoeff <= 1) and (0 <= darkcoeff <= 1):

        # Check if valid mode has been entered, then perform the right leak
        if mode == 'l':
            dimimg, lleak = LightLeak(src, darkcoeff, blendcoeff)
            result = np.hstack((dimimg, lleak))
            cv2.imshow('Light Leak', result)
            cv2.imwrite('LightLeak.jpg', lleak)
            cv2.waitKey(0)

        elif mode == 'r':
            dimimg, rleak = RainbowLeak(src, darkcoeff, blendcoeff)
            result = np.hstack((dimimg, rleak))
            cv2.imshow('Rainbow Leak', result)
            cv2.imwrite('RainbowLeak.jpg', rleak)
            cv2.waitKey(0)
        
        else:
            print('An invalid mode has been entered.') 
            print('Please choose l for light leak filter or r for rainbow leak filter.')
            print('example: IPCW.py p1 face1.jpg l 0.8')

    else:
        print('An invalid blending coefficient has been entered.')
        print('Please set the second parameter of problem1() as a number between 0 and 1 inclusive.')

def LightLeak(src, darkcoeff, blendcoeff):
    img = cv2.imread(src)
    mask = cv2.imread('BBm1.jpg')
    rows, cols, channels = img.shape

    dimimg = cv2.add(img, np.array([-100.0*darkcoeff]))
    brightimg = cv2.add(img, np.array([120.0]))

    roi = cv2.bitwise_and(brightimg, mask)
    diffuse_roi = cv2.GaussianBlur(roi, (11,11), 0, 0)
    res = cv2.addWeighted(dimimg, 1-blendcoeff, diffuse_roi, blendcoeff, 0)

    return dimimg, res

def RainbowLeak(src, darkcoeff, blendcoeff):
    img = cv2.imread(src)
    mask = cv2.imread('BBrm1.jpg')
    rows, cols, channels = img.shape

    dimimg = cv2.add(img, np.array([-100.0*darkcoeff]))
    
    rb = np.zeros(img.shape, np.uint8)
    # create it as an empty red image, since red is 0 hue angle
    rb[:] = (0, 0, 255)
    #convert to hsv to change hue
    rb_hsv = cv2.cvtColor(rb, cv2.COLOR_BGR2HSV)

    for y in range(rows):
        for x in range(int(cols*0.5), int(cols*0.625)):
            rb_hsv[y][x][0] = int(((x-cols*0.5)/(cols*0.125)) * 200)
    
    rb = cv2.cvtColor(rb_hsv, cv2.COLOR_HSV2BGR)
    #brighten rainbow because we dont want to use fin (aliasing artefacts)
    rb = cv2.add(rb, np.array([100.0]))

    roi = cv2.bitwise_and(rb, mask)
    diffuse_roi = cv2.GaussianBlur(roi, (31,31), 0, 0)
    res = cv2.addWeighted(dimimg, 1-blendcoeff, diffuse_roi, blendcoeff, 0)

    return dimimg, res


# excecuted on command line: 
# python main.py problem2 face1.jpg c 10 0.25
# motionblur controls stroke length
def problem2(src, mode, motionblur, blendcoeff):
    motionblur = int(motionblur)
    blendcoeff = float(blendcoeff)

    if motionblur >= 1 and 0<=blendcoeff<=1:
    
        if mode == 'm':
            result = monoSketch(src, motionblur, blendcoeff, gaussiannoise_sigma = 0.1)
            cv2.imshow('Monochrome Pencil Sketch filter', result)
            # Scale up image so it is visible in save file
            result = np.floor(255*result)
            cv2.imwrite('mono.jpg', result)
            cv2.waitKey(0)


        elif mode == 'c':
            result = colourSketch(src, motionblur, blendcoeff)
            cv2.imshow('Coloured Pencil Sketch filter', result)
            # Scale up image so it is visible in save file
            result = np.floor(result*255)
            cv2.imwrite('col.jpg', result)
            cv2.waitKey(0)

        else:
                print('An invalid mode has been entered.') 
                print('Please choose m for monochrome pencil sketch filter or c for coloured pencil sketch filter.')
                print('example: IPCW.py p2 face1.jpg m 10 0.25')

    else:
        print('Please use an integer motion blur parameter that is at least 1, and a blending coefficient between 0 and 1')



def monoSketch(src, motionblur, blendcoeff, gaussiannoise_sigma):

    img = cv2.imread(src)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape

    # generate noise texture
    noise = np.random.normal(0.5,1.25, img.shape)

    # Make motionblur kernel
    kernel = np.zeros((motionblur,motionblur))
    np.fill_diagonal(kernel, 1)
    kernel = kernel / motionblur # normalise kernel
    strokenoise = cv2.filter2D(noise, -1, kernel)

    # Apply dodging to grayscale image to look like a sketch
    inverse = 255 - img
    blurred = cv2.GaussianBlur(inverse, (21,21), 0, 0) # gaussian blur 21x21 kernel
    sketch = cv2.divide(img, 255 - blurred, scale=256)
    sketch = np.array(sketch/255, dtype=float) # must divide by 255 to normalise or else almost all white

    # blend image with stroke noise
    sketch = sketch*(1-blendcoeff) + strokenoise*blendcoeff

    return sketch


def colourSketch(src, motionblur, blendcoeff):
    img = cv2.imread(src)

    mono = monoSketch(src, motionblur, blendcoeff, gaussiannoise_sigma = 0.1)
    # get the negative of mono sketch, with pixel values in [0,1]
    inverse = 1 - mono

    # create a blank canvas
    blank = np.zeros(img.shape)    
    b, g, r = cv2.split(blank)

    # apply noise textures to two channels by inversion to achieve pencil effect
    g = inverse
    csketch = cv2.merge((b, g, r))

    # transfer sketch onto other two channels by inversion
    csketch = 1 - csketch


    return csketch


# This filter is called Smoothy Beauty
# enter python main.py problem3 face1.jpg 15 80 80
def problem3(src, nhood_size, colour_sigma, space_sigma):
    nhood_size = int(nhood_size)
    colour_sigma = int(colour_sigma)
    space_sigma = int(space_sigma)
    
    # check if paramters lie in valid ranges
    if 0 <= nhood_size <= 20:
        if 0<= colour_sigma <= 100 and 0 <= space_sigma <= 100:
            
            img = cv2.imread(src)
            smoothimg = cv2.bilateralFilter(img, nhood_size, colour_sigma, space_sigma)
            warmimg = warmUp(smoothimg)

            final = contrastHisto(warmimg)

            result = np.hstack((smoothimg, warmimg, final))

            cv2.imshow('Smoothy Beauty filter', result)
            cv2.imwrite('Smoothy.jpg', smoothimg)
            cv2.imwrite('WarmUp.jpg', warmimg)
            cv2.imwrite('SmoothyBeauty.jpg', final)
            cv2.waitKey(0)

        else:
            print('Parameters entered out of range. Please adhere to the following ranges:')
            print('0 <= standard deviation for filtering in colour space <= 100')
            print('0 <= standard deviation for filtering in space <= 100')

    else:
        print('Parameters entered out of range. Please adhere to the following ranges:')
        print('0 <= neighbourhood size <= 20')


def warmUp(src):
    increaseVals = populateLookupTable([0, 64, 128, 192, 255], [0, 80, 160, 230, 255])
    decreseVals = populateLookupTable([0, 64, 128, 192, 255], [0, 50, 100, 150, 255])

    b, g, r = cv2.split(src)
    r = cv2.LUT(r, increaseVals).astype(np.uint8)
    b = cv2.LUT(b, decreseVals).astype(np.uint8)

    return cv2.merge([b, g, r])

def populateLookupTable(x, y):
    spline = UnivariateSpline(x, y)
    return spline(range(256))


# Apply contrast limited adaptive histogram equalization to lightness channel in LAB space
def contrastHisto(src):
    img_lab= cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    newl_img = cv2.merge((cl,a,b))
    final = cv2.cvtColor(newl_img, cv2.COLOR_LAB2BGR)

    return final


# Run on command line using:
# IPsgwr64.py problem4 face2.jpg 0.7 200 170 BL 0
def problem4(src, strength, radius, lowpass_radius, interpolation_mode, inverse_swirl):
    strength = float(strength)
    radius = int(radius)
    lowpass_radius = int(lowpass_radius)
    inverse_swirl = int(inverse_swirl)

    img = cv2.imread(src)
    rows, cols = img.shape[:2]

    if 1 <= radius <= rows and 1 <= radius <= cols and 1 <= lowpass_radius <= rows and 1 <= lowpass_radius <= cols:

        if 0 <= strength and (inverse_swirl == 1 or inverse_swirl == 0):

            #PRE-FILTERING
            LPimg = lowPass(img, lowpass_radius)

            # Apply transformation by reverse mapping on destination
            outputimg = LPimg.copy()

            if interpolation_mode == 'NN':
                swirledimg = NearestNeighbour(LPimg, outputimg, rows, cols, strength, radius, inverse_swirl)
                inverseimg = swirledimg.copy()
                inverseimg = NearestNeighbour(swirledimg, inverseimg, rows, cols, strength, radius, 1-inverse_swirl)
                diff = cv2.subtract(inverseimg, img)
                result = np.hstack((LPimg, swirledimg,inverseimg, diff))
                cv2.imshow('Face Swirl filter with NN interpolation', result)
                cv2.imwrite('LowPass.jpg', LPimg)
                cv2.imwrite('Swirl.jpg', swirledimg)
                cv2.imwrite('InvSwirl.jpg', inverseimg)
                cv2.waitKey(0)


            elif interpolation_mode == 'BL':
                swirledimg = BiLinear(LPimg, outputimg, rows, cols, strength, radius, inverse_swirl)
                inverseimg = swirledimg.copy()
                inverseimg = BiLinear(swirledimg, inverseimg, rows, cols, strength, radius, 1-inverse_swirl)
                diff = cv2.subtract(inverseimg, img)
                result = np.hstack((LPimg, swirledimg,inverseimg, diff))
                cv2.imshow('Face Swirl filter with BL interpolation', result)
                cv2.imwrite('LowPass.jpg', LPimg)
                cv2.imwrite('Swirl.jpg', swirledimg)
                cv2.imwrite('InvSwirl.jpg', inverseimg)
                cv2.waitKey(0)

            else:
                print("Please enter a valid interpolation method.")
                print("Use 'NN' for Nearest Neighbour Interpolation")
                print("Use 'BL' for Bi-linear Interpolation")

        else:
            print('Please use a non-negative swirl strength and ensure inverse_swirl is set to 0 or 1')

    else:
        print('Please use a swirl and lowpass radius smaller than the dimensions of your input image.')


def lowPass(src, lowpass_radius):
    rows, cols, chans = src.shape
    
    # Separate image into channels to apply DFT to each
    b, g, r = cv2.split(src)
    channels = [b, g, r]

    # create mask
    mask = np.zeros((rows, cols, 2), dtype=np.uint8)
    circle_centre = (int(rows/2), int(cols/2))
    circle_radius = lowpass_radius
    circle_color = (255, 255, 255)
    circle_thickness = -1

    mask = cv2.circle(mask, circle_centre, circle_radius, circle_color, circle_thickness)
    Butterworth_mask = cv2.GaussianBlur(mask, (31,31), 0, 0)

    # apply DFT to blue channel
    dft = cv2.dft(np.float32(b), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # list to store channels after implementing Low Pass on each of them
    LPchannels = []

    for colour in channels:
        # apply DFT to blue channel
        dft = cv2.dft(np.float32(colour), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # this line is only if you want to plot out what the DFT looks like
        # magnitude_spectrum1 = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

        # apply mask to DFT for low-pass filtering
        fshift = dft_shift*Butterworth_mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # normalise back to 8 bits so we can show it using opencv
        min, max = np.amin(img_back, (0,1)), np.amax(img_back, (0,1))
        img_back = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        LPchannels.append(img_back)

    LPcol = cv2.merge(LPchannels)

    return LPcol


def NearestNeighbour(src, dest, rows, cols, strength, radius, inverse_swirl):
    centr = (int(cols/2), int(rows/2))

    # Apply reverse mapping to dest
    for y in range(rows):
        for x in range(cols):

            #Calculate polar coordinate of this destination pixel
            dx = x - centr[0]
            dy = y - centr[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)

            #Calculate amount of swirl to apply
            swirl_amount = 1.0 - (r/radius)

            if swirl_amount > 0.0:
                #Calculate angular displacement of corresponding pixel in input, phi
                phi = strength * swirl_amount * 2 * np.pi
                theta += phi * pow(-1, inverse_swirl)
                
                newX = np.cos(theta) * r
                newY = np.sin(theta) * r

                dest[y][x] = src[round(centr[1]+newY)][round(centr[0]+newX)]

    return dest


def BiLinear(src, dest, rows, cols, strength, radius, inverse_swirl):
    centr = (int(cols/2), int(rows/2))

    # Apply reverse mapping to dest
    for y in range(rows):
        for x in range(cols):

            #Calculate polar coordinate of this destination pixel
            dx = x - centr[0]
            dy = y - centr[1]
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)

            #Calculate amount of swirl to apply
            swirl_amount = 1.0 - (r/radius)

            if swirl_amount > 0.0:
                #Calculate angular displacement of corresponding pixel in input, phi
                phi = strength * swirl_amount * 2 * np.pi
                theta += phi * pow(-1, inverse_swirl)
                
                newX = np.cos(theta) * r + centr[0]
                newY = np.sin(theta) * r + centr[1]

                # Perform bilinear interpolation
                # get integer and fractional parts of pixel coordiantes
                Xi = int(newX)
                Xf = newX - Xi
                Yi = int(newY)
                Yf = newY - Yi

                XisuccLim = min(Xi+1, cols-1)
                YisuccLim = min(Yi+1, rows-1)

                output_colours = []

                for channel in range(3):
                    # get four pixels to do interpolation with
                    bottom_left = src[Yi][Xi][channel]
                    bottom_right = src[Yi][XisuccLim][channel]
                    top_left = src[YisuccLim][Xi][channel]
                    top_right = src[YisuccLim][XisuccLim][channel]

                    #interpolate at bottom
                    b = Xf * bottom_right + (1. - Xf) * bottom_left

                    # interpolate at top
                    t = Xf * top_right + (1. - Xf) * top_left

                    # interpolate between the two horizontal lines, in y direction
                    pxf = Yf * t + (1. - Yf) * b

                    output_colours.append(int(pxf + 0.5))

                dest[y][x] = src[int(newY)][int(newX)] # check if this position space mapping is right
                dest[y][x][0], dest[y][x][1], dest[y][x][2] = output_colours[0], output_colours[1], output_colours[2] # i think this is the right colour spac mapping

    return dest




if __name__ == "__main__" and len(sys.argv) > 1:
    if sys.argv[1] == 'problem1':
        problem1(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == 'problem2':
        problem2(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == 'problem3':
        problem3(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif sys.argv[1] == 'problem4':
        problem4(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])