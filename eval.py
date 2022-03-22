import numpy as np
import cv2
from skimage import feature as skif
from sklearn.externals import joblib

def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):
    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp,  normed=True, bins=max_bins, range=(0, max_bins))
    return hist

def features(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_h = lbp_histogram(image[:,:,0]) # y channel
    cb_h = lbp_histogram(image[:,:,1]) # cb channel
    cr_h = lbp_histogram(image[:,:,2]) # cr channel
    feature = np.concatenate((y_h,cb_h,cr_h))
    
    return np.array([feature])


def test(file_name):
    image = cv2.imread(file_name)
    image0s = image.copy()
    
    feature = features(image)

    model = joblib.load("./model.m")
    predict_proba = model.predict_proba(feature)
    predict = model.predict(feature)
    if predict[0] == 0:
        cv2.putText(image0s, "Fake", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    else:
        cv2.putText(image0s, "Real", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    print(predict)

    cv2.imshow("Result", image0s)
    cv2.waitKey(5000)
    


test("test.png")