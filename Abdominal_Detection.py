# -*- coding: utf-8 -*-
import cv2
import numpy as np
import copy

def red_detect(img):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([0,150,50])
    hsv_max = np.array([30,255,255])
    mask1 = cv2.inRange(hsv,hsv_min,hsv_max)

    hsv_min = np.array([150,150,50])
    hsv_max = np.array([179,255,255])
    mask2 = cv2.inRange(hsv,hsv_min,hsv_max)
    
    return mask1 + mask2

# 緑色の抽出
def green_detect(img):
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    hsv_min = np.array([30,50,50])
    hsv_max = np.array([90,255,255])
    mask1 = cv2.inRange(hsv,hsv_min,hsv_max)

    
    return mask1


def main():
    flag = 0
    #kernel = np.ones((3,3),np.uint8)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_EXPOSURE,-7.0)
    cap.set(cv2.CAP_PROP_CONTRAST,150.0)
    cap.set(cv2.CAP_PROP_SATURATION,128.0)

    while(cap.isOpened()):

        ret, frame = cap.read()
        # print("EXPOSURE:    {}",cap.get(cv2.CAP_PROP_EXPOSURE))
        # print("BRIGHTNESS:  {}",cap.get(cv2.CAP_PROP_BRIGHTNESS))
        # print("CONTRAST:  {}",cap.get(cv2.CAP_PROP_CONTRAST))
        # print("SATURATION:  {}",cap.get(cv2.CAP_PROP_SATURATION))

        h,w,c = frame.shape

        #mask = red_detect(frame)
        mask = green_detect(frame)

        mask = cv2.bitwise_not(mask)

        mask = mask[int((1/3)*h):int((2/3)*h),0:w]
        cv2.rectangle(frame,(0,int((1/3)*h)),(w,int((2/3)*h)),(0,0,255))
        cv2.imshow("mask",mask)

        # mask = mask[]
        label = cv2.connectedComponentsWithStats(mask)
        n = label[0] - 1
        data = np.delete(label[2],0,0)
        center = np.delete(label[3],0,0)

        color_src = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        S_data = data[:,4]
        top_S_data = S_data.max
        top_S_data = 0

        for i in range(n):
            if data[i][4] > 200:
                if top_S_data < data[i][4]:
                    top_idx = i
                    top_S_data = data[i][4]
                    x0 = data[i][0]
                    y0 = data[i][1]
                    x1 = data[i][0] + data[i][2]
                    y1 = data[i][1] + data[i][3]
                    cv2.rectangle(color_src, (x0, y0),(x1,y1),(0,0,255))

                    cv2.putText(color_src,"ID: " + str(i + 1), (x0 , y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))
                    cv2.putText(color_src,"S: " + str(data[i][4]), (x0 , y1 - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))

                    cv2.putText(color_src,"X: " + str(int(center[i][0])), (x1 -10 , y1 - 15), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))
                    cv2.putText(color_src,"Y: " + str(int(center[i][1])), (x1 -10 , y1 - 30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))
        
        # 腹式呼吸の状態判定
        if flag == 0:
            S = top_S_data
            flag = 1
        else:
            if S < top_S_data:
                state = 1
            elif S == top_S_data:
                state = 0
            elif S > top_S_data:
                state = -1
        
            S = top_S_data
            cv2.putText(color_src,"state:" + str(state),(0,20),cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255))
        # cv2.imshow("Frame",frame)
        cv2.imshow("color_src", color_src)
        # cv2.imshow("Mask",mask)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()