import numpy as np
import cv2
from sklearn.externals import joblib

def save_features(file_name):
    image = cv2.imread(file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_h = lbp_histogram(image[:,:,0]) # y channel
    cb_h = lbp_histogram(image[:,:,1]) # cb channel
    cr_h = lbp_histogram(image[:,:,2]) # cr channel
    feature = np.concatenate((y_h,cb_h,cr_h))
    
    return feature


def test(feature):
    model = joblib.load("./model.m")
    predict_proba = model.predict_proba(test_feature)
    predict = model.predict(feature)
    
    print("predict proba is:%f prediction is:%f"%(predict_proba,predict))


test("test.png")