import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)
capture2 = cv.VideoCapture(2)

ones = np.ones((480, 640), dtype = np.uint8)


MIN_MATCH_COUNT = 10

while True:
    ret, frame = capture.read()
    ret2, frame2 = capture2.read()    
    frame2 = cv.flip(frame2, 0)


    grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grapyframe2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)


    sift = cv.xfeatures2d.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(frame,None)
    kp2, des2 = sift.detectAndCompute(frame2,None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []

    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,d = frame.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        # frame2 = cv.polylines(frame2,[np.int32(dst)],True,255,3, cv.LINE_AA)

        frame = cv.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))

        mask = np.zeros(frame2.shape, dtype = np.uint8)
        cv.fillPoly(mask,[np.int32(dst)], (255,255,255))
        cv.bitwise_and(mask, frame2, frame2)


    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None


    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # blur = cv2.GaussianBlur(frame, (11,11), 0)
    # blur2 = cv2.GaussianBlur(frame2, (11,11), 0)

    # blurDiff = cv2.absdiff(blur, blur2)
    sortedDiff = cv.absdiff(frame, frame2)
    originDiff = cv.absdiff(grayframe, grapyframe2)

    ret,th1 = cv.threshold(sortedDiff,100,255,cv.THRESH_BINARY)

    if (ret and ret2):
        cv.imshow("frame1", frame)
        cv.imshow("frame2", frame2)

        # cv2.imshow("thresh1", th1)
        # cv2.imshow("thresh2", th2)
        # cv2.imshow("thresh3", th3)

        # cv2.imshow("blur1", blur)
        # cv2.imshow("blur2", blur2)

        # cv2.imshow("blurDiff", blurDiff)
        cv.imshow("sortedDiff", sortedDiff)
        cv.imshow("originDiff", originDiff)
        
        cv.imshow('thresh', th1)


    if cv.waitKey(1) > 0: 
        break

capture.release()
cv2.destroyAllWindows()