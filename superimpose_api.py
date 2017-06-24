import cv2
import numpy as np
import math
import copy


class Superimposition(object):
    def superimpose(self, jewel_image, user_image, angle, bodyx, bodyy, jewel_length, jewellery_type):
        if jewellery_type == 'earring':
            jewel_image = self.rescale(jewel_image, self.getscalingfactor(jewel_image, jewel_length, 3.0, 505.0))
            f = 4.5
            dz = 20
        elif jewellery_type == 'necklace':
            jewel_image = self.rescale(jewel_image, self.getscalingfactor(jewel_image, jewel_length, 25.0, 587.0))
            jewel_image = cv2.copyMakeBorder(jewel_image, 10, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            f = 1
            dz = 0
        img = cv2.cvtColor(jewel_image, cv2.COLOR_RGB2BGR)
        dst = self.rotate(img, 0, 1, dz, f, angle)
        dst = self.contour_centering(dst)
        xpos, ypos = self.get_leftmostpoint(dst)
        return self.superimpose_centered(dst, user_image, bodyx - xpos, bodyy - ypos)

    def rotate(self, img, dx, dy, dz, f, angle):
        beta = angle * 1. / 60.  # 0.5 & 0.7 &0.8 (left)
        beta = beta * 3.14 / 180.

        h, w, _ = img.shape

        # Projection 2D -> 3D matrix

        A1 = np.array([[1, 0, -w / 2.0], [0, 1, -h / 2.0], [0, 0, 1], [0, 0, 1]])

        R = np.array([[math.cos(beta), 0, -math.sin(beta), 0],
                      [0, 1, 0, 0],
                      [math.sin(beta), 0, math.cos(beta), 0],
                      [0, 0, 0, 1]])

        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz], [0, 0, 0, 1]])

        # 3D -> 2D matrix

        A2 = [[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]]

        # Final transformation matrix
        trans = (np.matmul(A2, np.matmul(T, np.matmul(R, A1))))

        dst = cv2.warpPerspective(img, trans, (w, h))
        return dst

    def contour_centering(self, dst):
        imgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        for i in range(imgray.shape[0]):
            for j in range(imgray.shape[1]):
                if imgray[i][j] > 245:
                    imgray[i][j] = 0
        im2, contours, hierarchy = cv2.findContours(imgray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        maxarea = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > maxarea:
                maxcnt = cnt
                maxarea = cv2.contourArea(cnt)

        x, y, w, h = cv2.boundingRect(maxcnt)
        dst = dst[y:y + h, x:x + w]
        return dst

    def superimpose_centered(self, dst, user_image, xpos, ypos):
        s_img = dst
        s_img_2 = cv2.cvtColor(s_img, cv2.COLOR_RGB2GRAY)
        l_img = user_image
        l_img = cv2.cvtColor(l_img, cv2.COLOR_RGB2BGR)
        l_img_2 = copy.deepcopy(l_img)
        x_offset = xpos
        y_offset = ypos
        l_img_2[y_offset:y_offset + s_img.shape[0], x_offset:x_offset + s_img.shape[1]] = s_img
        for i in range(y_offset, y_offset + s_img.shape[0]):
            for j in range(x_offset, x_offset + s_img.shape[1]):
                if s_img_2[i - y_offset][j - x_offset] < 10 or s_img_2[i - y_offset][j - x_offset] > 245:
                    l_img_2[i][j] = l_img[i][j]

        return l_img_2

    def get_leftmostpoint(self, jewel_image):
        jewel_gray = cv2.cvtColor(jewel_image, cv2.COLOR_RGB2GRAY)
        for i in range(jewel_gray.shape[0]):
            for j in range(jewel_gray.shape[1]):
                if jewel_gray[i][j] > 10 and jewel_gray[i][j] < 255:
                    return j, i

    def rescale(self, image, k):
        return cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)

    def getscalingfactor(self, jewel_image, length, model_length, model_pixel):
        jewel_gray = cv2.cvtColor(jewel_image, cv2.COLOR_RGB2GRAY)
        up = 0
        down = len(jewel_image)
        flag = False
        for i in range(5, jewel_gray.shape[0]):
            for j in range(5, jewel_gray.shape[1]):
                if not (jewel_gray[i][j] < 10 or jewel_gray[i][j] > 245):
                    up = i
                    flag = True
                    break
            if flag:
                break
        flag = False
        for i in range(jewel_gray.shape[0] - 5, 0, -1):
            for j in range(5, jewel_gray.shape[1]):
                if not (jewel_gray[i][j] < 10 or jewel_gray[i][j] > 245):
                    down = i
                    flag = True
                    break
            if flag:
                break

        scalingfactor = (model_pixel * length) / (model_length * (down - up))
        return scalingfactor