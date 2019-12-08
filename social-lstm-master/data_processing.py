import numpy as np
import pandas as pd

#data consists of 
#1 track ID
#2 xmin
#3 ymin
#4 xmax
#5 ymax
#6 frame
#7 lost
#8 occluded
#9 generated
#10 label

def preprocess_data():
    for i in range(15):
        df = pd.read_csv('selected_data/annotations{}.txt'.format(i), header=None, sep=' ')
        pedDF = df.loc[df[9] == 'Pedestrian'].drop([9], axis=1)
        pedDF = pedDF.loc[(pedDF[6] == 0)]
        pedDF = pedDF.loc[pedDF[7] == 0]
        pedDF = pedDF.drop([6, 7, 8], axis=1)
        pedMat = pedDF.to_numpy()
        x = (pedMat[:, 1] + pedMat[:, 3]) / 2.0
        y = (pedMat[:, 2] + pedMat[:, 4]) / 2.0
        frame = pedMat[:, 5]
        trackID = pedMat[:, 0]
        newIdx = np.argsort(frame)
        trackID = trackID[newIdx]
        x = x[newIdx]
        y = y[newIdx]
        output = np.concatenate(([frame], [trackID], [x], [y]), axis=0).T
        output[:,:2].astype('int')
        output[:,2:].astype('float')
        np.savetxt('processed_data/processed_data{}.txt'.format(i), output, delimiter=' ', fmt='%.1f')

def create_test_data():
    file_name = 'gates_8.txt'
    df = pd.read_csv(file_name, header=None, sep=' ')
    mat = df.to_numpy(dtype=str)
    row_length = mat.shape[0]
    traj_num = int((row_length+1) / 20)
    for i in range(traj_num):
        for j in range(8, 20):
            mat[i * 20 + j, 2:] = '?'
    np.savetxt('new_test_data.txt', mat, delimiter=' ', fmt='%s')
if __name__ == '__main__':
        create_test_data()
