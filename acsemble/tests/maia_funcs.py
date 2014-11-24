"""
Copyright (c) 2013, Matthew Dimmock
All rights reserved.


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
  * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
  * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
  * Neither the Australian Synchrotron nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os as os
import numpy as np
import pylab as plt
import itertools as it

def strings2Floats(dataLine):
    data = []
    #print dataLine
    for val in dataLine:
        data.append(float(val))
    return data    

def readData(fileName, toStrip, splitOn):
    # Run function as iterator.
    # Pass file descriptor.
    with open(fileName, 'rt') as fd:
        # Loop over lines in file.
        for line in fd:
            # Strip out white space.
            if toStrip:
                line = line.strip().strip()    
            parts = line.split(splitOn)
            yield parts, line

def listDirCont(dirPath, keepPath = False):
    # Return a lis of files in dirPath.
    if not keepPath:
        return os.listdir(dirPath)
    return [dirPath + "/" + fileName for fileName in os.listdir(dirPath)]

def filterDirCont(fileList, filtStr, sort = False):
    # Filter fileList according to filtStr.
    if sort:
        return sorted([fileName for fileName in fileList if filtStr in fileName])
    else:
        return [fileName for fileName in fileList if filtStr in fileName]


def getMaiaCollData(filePath):
    """
    Returns the collimator data for the Maia 384 pixel detector.
    """
    # Get the run number.
    collNum = int(os.path.basename(filePath).split('-')[-1].split('.csv')[0])
        
    # Get the data.
    data = list(readData(filePath, True, ','))
        
    # Remove the header lines.
    data = [strings2Floats(line[0]) for line in data if '#' not in line[0][0]]
    # Split the data into the components
    idXs, idYs, xLs, yLs, xHs, yHs = np.asarray(data).T
        
    # Declare empty list for collimator dimensions.
    collDims = []
        
#     plt.figure()
    
    for index, idX in enumerate(idXs):
       
        # X and Y positions of first quadrant.
        xP1 = (0.5 * (xHs[index] - xLs[index])) + xLs[index]
        yP1 = (0.5 * (yHs[index] - yLs[index])) + yLs[index]
        
        # X and Y lengths of first quadrant.
        xL1 = (xHs[index] - xLs[index])
        yL1 = (yHs[index] - yLs[index])
        
        # Append to list.
        collDims.append([xP1, yP1, xL1, yL1])
       
        # X and Y positions of second quadrant.
        # X coords are negative.
        xP2 = ((0.5 * (xHs[index] - xLs[index])) + xLs[index]) * -1.0
        yP2 = (0.5 * (yHs[index] - yLs[index])) + yLs[index]
        
        # Append to list.
        collDims.append([xP2, yP2, xL1, yL1])
        
        # X and Y positions of second quadrant.
        # Y coords are negative.
        xP3 = (0.5 * (xHs[index] - xLs[index])) + xLs[index]
        yP3 = ((0.5 * (yHs[index] - yLs[index])) + yLs[index]) * -1.0
        
        # Append to list.
        collDims.append([xP3, yP3, xL1, yL1])
        
        # X and Y positions of first quadrant.
        # Both X and Y are negative.
        xP4 = ((0.5 * (xHs[index] - xLs[index])) + xLs[index]) * -1.0
        yP4 = ((0.5 * (yHs[index] - yLs[index])) + yLs[index]) * -1.0
        
        # Append to list.
        collDims.append([xP4, yP4, xL1, yL1])
        
#         plt.scatter(xLs[index], yLs[index], color='r')
#         plt.scatter(xHs[index], yHs[index], color='b')
#         plt.scatter(xP1, yP1, color='g')
#         plt.scatter(xP2, yP2, color='g')
#         plt.scatter(xP3, yP3, color='g')
#         plt.scatter(xP4, yP4, color='g')
#         plt.xlim([-10.0, 10.0])
#         plt.ylim([-10.0, 10.0])

    return collDims


def getCollData(filePath, flag):
    """
    Control function for returning the data for each of the two XFM detectors.
    """
    collDims = None
    
    # Get the collimator data for the Maia 384 pixel detector.
    if flag == "maia":
        collDims = getMaiaCollData(filePath)
    
    return collDims

def getPixelMap():
    """
    Returns a pixel map for the channels of the Maia detector.
    The vertical axis is the y axis and the horizontal is z in Geant space.
    The pixel indices start from the bottom left corner of this map.
    """
    pixelMap = [[295, 294, 293, 292, 291, 290, 289, 288, 88 , 80 , 72 , 64 , 63 , 48 , 47 , 32 , 31 , 23 , 15 , 7  ],
                [303, 302, 301, 300, 299, 298, 297, 296, 89 , 81 , 73 , 65 , 62 , 55 , 40 , 33 , 30 , 22 , 14 , 6  ],
                [311, 310, 309, 308, 307, 306, 305, 304, 90 , 82 , 74 , 66 , 61 , 54 , 41 , 34 , 29 , 21 , 13 , 5  ],
                [319, 318, 317, 316, 315, 314, 313, 312, 91 , 83 , 75 , 67 , 60 , 53 , 42 , 35 , 28 , 20 , 12 , 4  ],
                [320, 321, 322, 323, 324, 325, 326, 327, 92 , 84 , 76 , 68 , 59 , 52 , 43 , 36 , 27 , 19 , 11 , 3  ],
                [335, 328, 329, 330, 331, 332, 333, 334, 93 , 85 , 77 , 69 , 58 , 51 , 44 , 37 , 26 , 18 , 10 , 2  ],
                [336, 343, 342, 341, 340, 339, 338, 337, 94 , 86 , 78 , 70 , 57 , 50 , 45 , 38 , 25 , 17 , 9  , 1  ],
                [351, 350, 349, 348, 347, 346, 345, 344, 95 , 87 , 79 , 71 , 56 , 49 , 46 , 39 , 24 , 16 , 8  , 0  ],
                [352, 353, 354, 355, 356, 357, 358, 359, -1 , -1 , -1 , -1 , 191, 190, 189, 188, 187, 186, 185, 184],
                [360, 361, 362, 363, 364, 365, 366, 367, -1 , -1 , -1 , -1 , 183, 182, 181, 180, 179, 178, 177, 176],
                [368, 369, 370, 371, 372, 373, 374, 375, -1 , -1 , -1 , -1 , 175, 174, 173, 172, 171, 170, 169, 168],
                [376, 377, 378, 379, 380, 381, 382, 383, -1 , -1 , -1 , -1 , 167, 166, 165, 164, 163, 162, 161, 160],
                [192, 200, 208, 216, 231, 238, 241, 248, 263, 271, 279, 287, 152, 153, 154, 155, 156, 157, 158, 159],
                [193, 201, 209, 217, 230, 237, 242, 249, 262, 270, 278, 286, 145, 146, 147, 148, 149, 150, 151, 144],
                [194, 202, 210, 218, 229, 236, 243, 250, 261, 269, 277, 285, 142, 141, 140, 139, 138, 137, 136, 143],
                [195, 203, 211, 219, 228, 235, 244, 251, 260, 268, 276, 284, 135, 134, 133, 132, 131, 130, 129, 128],
                [196, 204, 212, 220, 227, 234, 245, 252, 259, 267, 275, 283, 120, 121, 122, 123, 124, 125, 126, 127],
                [197, 205, 213, 221, 226, 233, 246, 253, 258, 266, 274, 282, 112, 113, 114, 115, 116, 117, 118, 119],
                [198, 206, 214, 222, 225, 232, 247, 254, 257, 265, 273, 281, 104, 105, 106, 107, 108, 109, 110, 111],
                [199, 207, 215, 223, 224, 239, 240, 255, 256, 264, 272, 280, 96 , 97 , 98 , 99 , 100, 101, 102, 103]]
    return pixelMap

def getPixelTupleFromPosTuple(z_mm, y_mm):
    """
    Get the bin location for the histogram.
    Up until now, we have used x and y.  In Geant, these are y and z. 
    """
    
    # Get the centre of each pixel in the y,z-direction.
    pixCentres = np.arange(20) - 10.0 + 0.5
    
    # Get the difference between the coordinate and the centres of the pixels. 
    yDiffs = np.abs(y_mm - pixCentres)
    
    # The correct pixel is the minimum of these differences.
    
    yPixel = list(yDiffs).index(np.min(np.abs(yDiffs)))
    
    # Get the difference between the coordinate and the centres of the pixels. 
    zDiffs = np.abs(z_mm - pixCentres)
    
    # The correct pixel is the minimum of these differences.
    zPixel = list(zDiffs).index(np.min(np.abs(zDiffs)))
    
    return yPixel, zPixel

def getTupleFromChannel(pixelMap, chan):
    """
    Returns a tuple of the row and column for a given Maia detector channel.
    """
    # Get flattened pixel map.
    flatPixMap = list(it.chain.from_iterable(pixelMap))
    
    # Get the channel index.
    index = int(flatPixMap.index(chan))
    
    # Get the number of columns.
    numCols = int(len(pixelMap[0]))
    
    # Get the row.
    rowNum = abs(int(np.floor(float(index) / float(numCols))) - 19)
    
    # Get the column.
    colNum = int(index % numCols) 
    
    # Return the tuple from the channel.
    return (rowNum, colNum)

def getSolidAngle(a, b, d):
    """
    Get the solid angle.
    """
    
    alpha = a / (2.0 * d)
    
    beta = b / (2.0 * d)

    fact = np.sqrt((1 + alpha**2 + beta**2) / ((1 + alpha**2) * (1 + beta **2)))

    omega = 4.0 * (np.arccos(fact))
    
    return omega

def getChannelFromTuple(pixelMap, pixelTuple):
    """
    Returns the Maia channel number from a look-up table.
    """
    
    # The y index has to start from the bottom and not the top and so has to be inverted.
    yPix = abs(pixelTuple[0] - 19)
    
    # The z index stays the same.
    zPix = pixelTuple[1]
    
    # Return the channel from the tuple.    
    return pixelMap[yPix][zPix]

def getRadialDistance(yPix, zPix):
    """
    Get the radial distance to each pixel.
    yPix and zPix are along the Geant axis directions. 
    """
    # Get the centre of each pixel in the y,z-direction.
    pixCentres = np.arange(20) - 10.0 + 0.5
    
    # Translate the y pixel into a location.
    y_mm = pixCentres[yPix]
    
    # Translate the z pixel into a location.
    z_mm = pixCentres[zPix]
    
    return np.sqrt(y_mm**2 + z_mm**2)

def getCollAreas(dirPath, csvPath, dit2DetCent_mm, detThick_mm):
    """
    Get the area subtended by the collimator.
    """
    
    ########################
    # Get collimator areas #
    ########################
        
    collDims = getCollData(csvPath, "maia")
    
    # Initialize empty array to store area.
    areaArray = np.zeros([20,20])
    
    # Initialize empty array to store radii.
    radialArray = np.zeros([20,20])
    
    # Initialize empty array to store area.
    solidAngleArray = np.zeros([20,20])
    
    # Get the mapping of the channel numbers.    
    pixelMap = getPixelMap()
    
    # Initialize an empty solid angle list.
    radialSolidList = []
    
    # plt.figure()
    
    for dim in collDims:
        
        yPix, zPix = getPixelTupleFromPosTuple(dim[0], dim[1])
        
        # Get the corresponding experimental channel number from the pixel tuple.
        expChan = getChannelFromTuple(pixelMap, (yPix, zPix))
        
        # Get the radial distance to each pixel.
        radDist_mm = getRadialDistance(yPix, zPix)
        
        # Calculate the area.
        area_mm2 = dim[2] * dim[3]
        
        # Store the area.
        areaArray[yPix, zPix] = area_mm2
    
        # Store the radius.
        radialArray[yPix, zPix] = radDist_mm
        
        # Distance to collimator layer.
        dist2Coll_mm = dit2DetCent_mm - (0.5 * detThick_mm) - 0.3
          
        # Define distance to bottom corner.
        A = np.abs(dim[0]) - (0.5 * np.abs(dim[2]))
        B = np.abs(dim[1]) - (0.5 * np.abs(dim[3]))
        a = dim[2]
        b = dim[3]
         
        omega1 = getSolidAngle(2.0 * (A + a), 2.0 * (B + b), dist2Coll_mm)
        omega2 = getSolidAngle(2.0 * (A), 2.0 * (B + b), dist2Coll_mm)
        omega3 = getSolidAngle(2.0 * (A + a), 2.0 * (B), dist2Coll_mm)
        omega4 = getSolidAngle(2.0 * (A), 2.0 * (B), dist2Coll_mm)
        
        omega = (omega1 - omega2 - omega3 + omega4) / 4.0
        
        # Store the solid angle.
        solidAngleArray[yPix, zPix] = omega
        
        radialSolidList.append([radDist_mm, omega])
        
        # plt.scatter(radDist_mm, omega)
        # plt.xlabel('radDist_mm')
        # plt.ylabel('$\Omega$')

    # Transpose the list of solid angles.
    radialSolidArray = np.asarray(radialSolidList).T
    
    # Fit the solid angle function.
    solidFit = np.polyfit(radialSolidArray[0], radialSolidArray[1], 2)
    
    # Write the solidFit data to file.
    # Open a file to write the solidFit params to.
    fileName = "SolidAnglePolyFit_%.03fmm.xml" %(dit2DetCent_mm)
    solidFitParamsPath = os.path.join(dirPath, fileName)
    #~ solidFitParamsFile = open(solidFitParamsPath, 'w')
    
    # start writing to xml file
    lineOut = '<?xml version="1.0" encoding="UTF-8"?>\n\n'
    #~ solidFitParamsFile.write(lineOut)
    lineOut = '<polyFitParams>\n'
    #~ solidFitParamsFile.write(lineOut)
    
    for index, param in enumerate(solidFit):
        # Write the params to file.
        lineOut = '    <fit order="%d" value="%s">\n' %(index, param) 
        #~ solidFitParamsFile.write(lineOut)
        lineOut = '    </fit>\n'
        #~ solidFitParamsFile.write(lineOut)
    
    # Write the final line of XML to the params file.
    lineOut = '</polyFitParams>\n'
    #~ solidFitParamsFile.write(lineOut)
   
    # Close the XML file that has been written to.
    #~ solidFitParamsFile.close()
 
    # plt.figure()
    # plt.imshow(areaArray, interpolation = 'nearest', origin = 'lower', cmap=plt.cm.hot)
    # plt.title('area $mm^2$')
    # plt.colorbar()
    
    # plt.figure()
    # plt.imshow(radialArray, interpolation = 'nearest', origin = 'lower', cmap=plt.cm.hot)
    # plt.title('radial dist mm')
    # plt.colorbar()
    
    return solidAngleArray, areaArray

if __name__ == "__main__":
    
    # Distance from the centre of the foil to the detector.
    foil2DetCent_mm = 10.0
    
    # Shape of detector.
    detShape_mm = np.asarray([0.4, 20.0, 20.0])
    
    # Specify the directory path to read from and write to.
    dirPath = "."
    
    # Location of CSV file that has the collimator info in it.
    csvPath = os.path.join(dirPath, "mask-Mo-3x100-R-10-T-04-gap-75-grow-09-tune-103-safe-01-glue-025-1.csv")
    
    # Get the solid angle distribution.
    solidAngleArray = getCollAreas(dirPath, csvPath, foil2DetCent_mm, detShape_mm[0])
    
    print "Done."
    # plt.show()
    