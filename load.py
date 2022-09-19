import numpy as np
from scipy import misc

def load_data(data, frames, batch_size, Height, Width, Channel, folder,forward_flag):
    for b in range(batch_size):
        path = folder[np.random.randint(len(folder))] + '/'
        bb = np.random.randint(0, 447 - 256)
        if (forward_flag%2 == 0):
            for f in range(frames):
                if f == 0:
                    img = misc.imread(path + 'im1_CA_6' + '.png')
                    #img = misc.imread(path + 'im' + str(f + 1) + '.png')
                    data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
                else:
                    img = misc.imread(path + 'im' + str(f + 1) + '.png')
                    data[f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
        else:
            for f in range(frames):
                if f == 6:
                    img = misc.imread(path + 'im7_CA_3' + '.png')
                    data[6-f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]
                else:
                    img = misc.imread(path + 'im' + str(f + 1) + '.png')
                    data[6-f, b, 0:Height, 0:Width, 0:Channel] = img[0:Height, bb: bb + Width, 0:Channel]

    return data
