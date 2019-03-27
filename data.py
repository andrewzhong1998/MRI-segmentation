import nrrd
import os
import numpy as np
def load_dataset():
    pretrain_x1 = np.zeros(shape=(22,64,64,64))
    pretrain_x2 = np.zeros(shape=(22,64,64,64))
    pretrain_x = np.zeros(shape=(22,64,64,64,2))
    pretrain_y = np.zeros(shape=(22,64,64,64))
    pretrain_dir_local = "/Users/andrew/Desktop/research2019/1MonthPretrainingData"
    pretrain_dir_remote = "/proj/NIRAL/users/mingzhi/1MonthPretrainingData"
    pretrain_dir = pretrain_dir_local
    pretrain_files = os.listdir(pretrain_dir)
    pretrain_files.sort()
    for i in range(0,22):
        print("Instance "+str(i+1)+" loaded")

        x1, h1 = nrrd.read(pretrain_dir+"/"+pretrain_files[3*i])
        x1 = np.pad(x1, ((0, 0), (0,0), (3, 3)), 'constant')
        x1 = x1[22:278, 47:303, :]
        x1 = x1[::4, ::4, ::4]

        x2, h3 = nrrd.read(pretrain_dir + "/" + pretrain_files[3*i+2])
        x2 = np.pad(x2, ((0, 0), (0, 0), (3, 3)), 'constant')
        x2 = x2[22:278, 47:303, :]
        x2 = x2[::4, ::4, ::4]

        x = np.stack((x1,x2),axis=-1)

        y, h = nrrd.read(pretrain_dir + "/" + pretrain_files[3*i+1])
        y = np.pad(y, ((0, 0), (0, 0), (3, 3)), 'constant')
        y = y[22:278, 47:303, :]
        y = y[::4,::4,::4]

        pretrain_x1[i] = x1
        pretrain_x2[i] = x2
        pretrain_x[i] = x
        pretrain_y[i] = y

    print("Pretrain sets loaded completely")

    return pretrain_x1, pretrain_x2, pretrain_x, pretrain_y

#pretrain_x1_resized, pretrain_x2_resized, pretrain_x_resized, pretrain_y_resized = load_dataset()
#print(pretrain_y_resized.shape)
labels = [0.0, 1.0, 2.0, 3.0]
l = len(labels)
ret = np.zeros(l)
for i in range(l):
    label = labels[i]
    t = pretrain_y_resized == label
    ret[i] = t.sum()
print(ret)
#print(pretrain_x1_resized.shape)
#np.save("/Users/andrew/Desktop/npdata64/pretrain_x1_resized.npy", pretrain_x1_resized)
#np.save("/Users/andrew/Desktop/npdata64/pretrain_x2_resized.npy", pretrain_x2_resized)
#np.save("/Users/andrew/Desktop/npdata64/pretrain_x_resized.npy", pretrain_x_resized)
#np.save("/Users/andrew/Desktop/npdata64/pretrain_y_resized.npy", pretrain_y_resized)
