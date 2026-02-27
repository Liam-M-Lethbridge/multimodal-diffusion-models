import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



def getRandomBatch(batchSize:int, location:str):
    directory = "../../data/LOCATION/"
    directory= directory.replace("LOCATION", location)
    names = [f for f in os.listdir(directory+"RGB/") if f.endswith(".npy")]
    indices = np.random.uniform(0, len(names), batchSize).astype(int)

    RGBs = []
    SEGs = []
    for index in indices:
        name = names[index]
        RGBs.append(np.load(f"{directory}/RGB/{name}"))
        SEGs.append(tf.one_hot(np.load(f"{directory}/SEG/{name}")//10, 10))
    print(RGBs)
    return RGBs, SEGs

def getBatchInOrder(batchSize:int, location:str, startIndex:int):
    directory = "../../data/LOCATION/"
    directory= directory.replace("LOCATION", location)
    names = [f for f in os.listdir(directory+"RGB/") if f.endswith(".npy")]

    RGBs = []
    SEGs = []
    for i in range(batchSize):
        if startIndex + i >= len(names):
            print("FUCK YOU")
            break
        name = names[startIndex+i]
        RGBs.append(np.load(f"{directory}/RGB/{name}"))
        SEGs.append(tf.one_hot(np.load(f"{directory}/SEG/{name}")//10, 10))
    return RGBs, SEGs, i+1


if __name__ == "__main__":
    rgbs, segs, i = getBatchInOrder(1, "UK",0)
    # fig, axes = plt.subplots(1,2)
    # axes[0].imshow(rgbs[0]*3.5)
    print(segs[0])
    plt.imshow(rgbs[0]*3.5)
    plt.show()
