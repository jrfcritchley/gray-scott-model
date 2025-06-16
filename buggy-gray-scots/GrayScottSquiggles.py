import cv2  # Not actually necessary if you just want to create an image.
import numpy as np
import random

height = 200
width = 200
aDiffRate = 1
bDiffRate = 0.5
feedRate = 0.192
killRate = 0.192
dT =2 # 2 makes bizare spirally dna patterns
dx = 1
aChem = 0
bChem = 0
initBConc = 8
epochs = 360
avgConc = 0

#change this number to like 20 for cool patterns
Breakdown = 1


#0.192 works well for rates
#F=0.0620, k=0.0630.
#Here are two videos that show a "mitosis" simulation (f=.0367, k=.0649) and a "coral growth" simulation (f=.0545, k=.062)
aConcColour = 0
bConcColour = 0


aConc = np.ones((height,width), dtype=np.double)


bConc = np.zeros((height,width), dtype=np.double)

##for i in range(initBConc): #This is the number of time an initial amount of chemical B is placed
##    bConc[random.randint(10,width-10),random.randint(10,height-10)] = 1

bConc[100,100] = 1


combConc = np.zeros((height,width,3), dtype=np.double)

def num_to_range(num, inMin, inMax, outMin, outMax):
  return outMin + (float(num - inMin) / float(inMax - inMin) * (outMax - outMin))

def Laplace(loc0,loc1,loc2,loc3,loc4,loc5,loc6,loc7,loc8):
    lap = ((loc0*0.05) + (loc1*0.2) + (loc2*0.05) + (loc3*0.2) + (loc4*-1) + (loc5*0.2) + (loc6*0.05) + (loc7*0.2) + (loc8*0.05))
    return lap*Breakdown

for reps in range(epochs):
    for i in range(1,height-1):
        for j in range(1,width-1):

            aChem = aConc[i, j] + ((aDiffRate * Laplace(aConc[i-1,j-1],aConc[i-1,j],aConc[i-1,j+1],aConc[i,j-1],aConc[i,j],aConc[i,j+1],aConc[i+1,j-1],aConc[i+1,j],aConc[i+1,j+1]) - (aConc[i, j]* (bConc[i,j]**2))) + (feedRate * (1 - aConc[i, j])) * dT)
            bChem = bConc[i,j] + ((bDiffRate * Laplace(bConc[i-1,j-1],bConc[i-1,j],bConc[i-1,j+1],bConc[i,j-1],bConc[i,j],bConc[i,j+1],bConc[i+1,j-1],bConc[i+1,j],bConc[i+1,j+1]) + (aConc[i, j]* (bConc[i,j]*2))) - ((killRate + feedRate)* bConc[i,j]) * dT)

            aConc[i, j] = max(0, min(1, aChem))
            bConc[i, j] = max(0, min(1, bChem))


                
    aConc_scaled = (aConc * 255).astype(np.uint8)
    bConc_scaled = (bConc * 255).astype(np.uint8)

    blue_channel = np.zeros_like(aConc_scaled, dtype=np.uint8)
    red_channel = np.zeros_like(bConc_scaled, dtype=np.uint8)

    blue_channel[:] = aConc_scaled
    red_channel[:] = bConc_scaled

    purple_channel = cv2.addWeighted(blue_channel, 0.5, red_channel, 0.5, 0)

    final_image = cv2.merge((blue_channel, purple_channel, red_channel))
    print("1")

    cv2.imwrite(f'C:/Users/Jack Critchley/Desktop/GrayScott/saved_img_{reps}.png', final_image) 

