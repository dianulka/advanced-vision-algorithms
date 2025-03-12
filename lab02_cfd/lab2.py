import cv2
import matplotlib.pyplot as plt
import numpy as np


# Zadanie 1.
# for i in range (300 ,1100) :
#     I = cv2.imread('pedestrian/input/in%06d.jpg' % i)
#
#     cv2.imshow("I", I)
#     cv2.waitKey(10)

def detection(folder_name, binary_threshold):
    f= open(f'{folder_name}/temporalROI.txt', 'r')
    line = f.readline()
    roi_start, roi_end = line.split()
    roi_start = int(roi_start)
    roi_end = int(roi_end)
    f.close()

    TP, TN, FP, FN = 0, 0, 0, 0
    prev = cv2.imread(f'{folder_name}/input/in000{roi_start}.jpg', cv2.IMREAD_GRAYSCALE).astype('int')

    for i in range (roi_start+1 ,roi_end+1) :
        I = cv2.imread(f'{folder_name}/input/in%06d.jpg' % i, cv2.IMREAD_GRAYSCALE).astype('int')

        diff = cv2.absdiff(I, prev).astype(np.uint8)

        (T, b) = cv2.threshold(diff, binary_threshold, 255, cv2.THRESH_BINARY)

        median = cv2.medianBlur(b, 9)

        erosion = cv2.erode(median, np.ones((3,3)), iterations = 1 )
        dilation = cv2.dilate(erosion, np.ones((3,3)), iterations = 1)


        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(dilation)
        print(f"Total number of unique labels in {i} frame: {retval} ")
        # retval -- total number of unique labels
        # labels -- destination labeled image, mapa obiektów gdzie każdemu pikselowi przpisano numer obiektu
        # stats -- statistics output for each label , including the background label .
        # centroids -- centroid output for each label , including the background label .

        cv2.imshow(" Labels ", np.uint8(labels / retval * 255))
        cv2.waitKey(10)
        prev = I
        I_VIS= cv2.imread(f'{folder_name}/input/in%06d.jpg' % i)
        if (stats.shape[0] > 1): # Threse is an object

            tab = stats[1:, 4]
            pi = np.argmax(tab)
            pi = pi + 1

            start_point = (stats[pi,0], stats[pi,1])
            end_point = (stats[pi,0]+stats[pi,2], stats[pi,1]+stats[pi,3])
            cv2.rectangle(img = I_VIS,pt1= start_point, pt2= end_point, color=(0,255, 0), thickness=2)

            cv2.putText(I_VIS, "%f0" % stats[pi,4], (stats[pi,0], stats[pi,1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(I_VIS, "%d" %pi, (np.int_(centroids[pi,0]), np.int_(centroids[pi,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        cv2.imshow("I_VIS", I_VIS)

        if roi_start <= i <= roi_end:
            B = cv2.imread(f'{folder_name}/groundtruth/gt%06d.png' % i, cv2.IMREAD_GRAYSCALE)
            TP += np.sum(np.logical_and(B == 255, dilation == 255))
            TN += np.sum(np.logical_and(B != 255, dilation != 255))
            FP += np.sum(np.logical_and(B != 255, dilation == 255))
            FN += np.sum(np.logical_and(B == 255, dilation != 0))



    P = TP / (TP + FP)
    R = TP / (TP + FN)
    F1 = (2 * P * R) / (P + R)

    print(f"{folder_name}")
    print(f"Precision: {P}")
    print(f"Recall: {R}")
    print(f"F1: {F1}")

    cv2.destroyAllWindows()
    return P, R, F1




results = {}
folder_names = ["pedestrian", "office", "highway"]
binary_thresholds = [10, 7, 13]
for folder_name, binary_threshold in zip(folder_names, binary_thresholds):
    results[folder_name] = {}
    P,R,F1 =  detection(folder_name, binary_threshold)

    results[folder_name]['P'] = float(P)
    results[folder_name]['R'] = float(R)
    results[folder_name]['F1'] = float(F1)

print (results)