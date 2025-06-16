import cv2
import numpy as np
import random

#simulation settings
height = 200
width = 200
aDiffRate = 1
bDiffRate = 0.5
feedRate = 0.055
killRate = 0.062
dT = 1

epochs = 10000

#initialize concentrations
aConc = np.ones((height, width), dtype=np.double)
bConc = np.zeros((height, width), dtype=np.double)

#noise to break symmetry
aConc += 0.01 * np.random.random((height, width))
bConc += 0.001 * np.random.random((height, width))

#seed chemical B 
for seed in range(20):
    x = random.randint(10, width - 10)
    y = random.randint(10, height - 10)
    bConc[y-5:y+5, x-5:x+5] = 1

#laplace diffusion kernel
def Laplace(loc0, loc1, loc2, loc3, loc4, loc5, loc6, loc7, loc8):
    lap = (
        (loc0 * 0.05) + (loc1 * 0.2) + (loc2 * 0.05) +
        (loc3 * 0.2)  + (loc4 * -1) + (loc5 * 0.2) +
        (loc6 * 0.05) + (loc7 * 0.2) + (loc8 * 0.05)
    )
    return lap * 1  #if you change  the 1 it breaks in cool ways sometimes

# Main simulation loop
for reps in range(epochs):
    aNext = np.copy(aConc)
    bNext = np.copy(bConc)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            lapA = Laplace(
                aConc[i-1,j-1], aConc[i-1,j], aConc[i-1,j+1],
                aConc[i,j-1],   aConc[i,j],   aConc[i,j+1],
                aConc[i+1,j-1], aConc[i+1,j], aConc[i+1,j+1]
            )

            lapB = Laplace(
                bConc[i-1,j-1], bConc[i-1,j], bConc[i-1,j+1],
                bConc[i,j-1],   bConc[i,j],   bConc[i,j+1],
                bConc[i+1,j-1], bConc[i+1,j], bConc[i+1,j+1]
            )

            a = aConc[i, j]
            b = bConc[i, j]

            aChem = a + (aDiffRate * lapA - a * b * b + feedRate * (1 - a)) * dT
            bChem = b + (bDiffRate * lapB + a * b * b - (killRate + feedRate) * b) * dT

            aNext[i, j] = np.clip(aChem, 0, 1)
            bNext[i, j] = np.clip(bChem, 0, 1)

    aConc = aNext
    bConc = bNext

    #image generation
    aConc_scaled = (aConc * 255).astype(np.uint8)
    bConc_scaled = (bConc * 255).astype(np.uint8)

    blue_channel = np.zeros_like(aConc_scaled, dtype=np.uint8)
    red_channel = np.zeros_like(bConc_scaled, dtype=np.uint8)

    blue_channel[:] = aConc_scaled
    red_channel[:] = bConc_scaled

    purple_channel = cv2.addWeighted(blue_channel, 0.5, red_channel, 0.5, 0)
    final_image = cv2.merge((blue_channel, purple_channel, red_channel))

    #printing every 10
    if reps % 10 == 0:
        cv2.imwrite(f'D:/Users/Jack Critchley/Desktop/GrayScott/saved_img_{reps}.png', final_image)
        print(f"Saved frame {reps}")
