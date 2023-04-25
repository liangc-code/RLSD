import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

def get_ssim(x, y):
    """
    :param x: input
    :param y: reference
    :return: SSIM
    """
    if x.ndim == 4:
        ssim_list = []
        data_len = len(x)
        for k in range(data_len):
            img = x[k,0]
            ref = y[k,0]
            ssim_value = ssim(ref, img, full=True)
            ssim_list.append(ssim_value[1])
        return abs(np.array(ssim_list))
    elif x.ndim == 2:
        ssim_value = ssim(y, x, full=True)
        return ssim_value[1].astype('float32')

class State_mix():
    def __init__(self, size, move_range):
        self.image = np.zeros(size,dtype=np.float32)
        self.move_range = move_range
    
    def reset(self, x, n):
        self.ori = n
        self.image = n
        size = self.image.shape
        prev_state = np.zeros((size[0], 64, size[2], size[3]), dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        self.image = x
        self.tensor[:,:self.image.shape[1],:,:] = self.image

    def step(self, act, inner_state):
        moved_image = self.image
        box = np.zeros(self.image.shape, self.image.dtype)
        box0 = np.zeros(self.image.shape, self.image.dtype)
        somf = np.zeros(self.image.shape, self.image.dtype)
        enhance = np.zeros(self.image.shape, self.image.dtype)
        dip = np.zeros(act.shape, act.dtype)
        dip = np.where(act[:, :, :] == self.move_range + 1, -2, dip)
        dip = np.where(act[:, :, :] == self.move_range + 2, -4, dip)
        dip = np.where(act[:, :, :] == self.move_range + 3, -6, dip)
        dip = np.where(act[:, :, :] == self.move_range + 4, -8, dip)
        dip = np.where(act[:, :, :] == self.move_range + 5, -10, dip)
        dip = np.where(act[:, :, :] == self.move_range + 6, -12, dip)
        dip = np.where(act[:, :, :] == self.move_range + 7, -14, dip)
        dip = np.where(act[:, :, :] == self.move_range + 8, 2, dip)
        dip = np.where(act[:, :, :] == self.move_range + 9, 4, dip)
        dip = np.where(act[:, :, :] == self.move_range + 10, 6, dip)
        dip = np.where(act[:, :, :] == self.move_range + 11, 8, dip)
        dip = np.where(act[:, :, :] == self.move_range + 12, 10, dip)
        dip = np.where(act[:, :, :] == self.move_range + 13, 12, dip)
        dip = np.where(act[:, :, :] == self.move_range + 14, 14, dip)
        b, c, h, w = self.image.shape
        for i in range(0,b):
            if np.sum(act[i]==self.move_range-2) > 0:
                box[i, 0] = cv2.boxFilter(self.image[i, 0], ddepth=-1, ksize=(9, 9))
            if np.sum(act[i]==self.move_range-1) > 0:
                box0[i, 0] = cv2.boxFilter(self.image[i, 0], ddepth=-1, ksize=(3, 3))
            if (np.sum(act[i] == self.move_range) > 0) or (np.sum(act[i] == self.move_range + 1) > 0) or \
                    (np.sum(act[i] == self.move_range + 2) > 0) or (np.sum(act[i] == self.move_range + 3) > 0) or \
                    (np.sum(act[i] == self.move_range + 4) > 0) or (np.sum(act[i] == self.move_range + 5) > 0) or \
                    (np.sum(act[i] == self.move_range + 6) > 0) or (np.sum(act[i] == self.move_range + 7) > 0) or \
                    (np.sum(act[i] == self.move_range + 8) > 0) or (np.sum(act[i] == self.move_range + 9) > 0) or \
                    (np.sum(act[i] == self.move_range + 10) > 0) or (np.sum(act[i] == self.move_range + 11) > 0) or \
                    (np.sum(act[i] == self.move_range + 12) > 0) or (np.sum(act[i] == self.move_range + 13) > 0) or \
                    (np.sum(act[i] == self.move_range + 14) > 0):
                utmp = np.zeros((h * w, 9))
                utmp[:, 4] = self.image[i, 0].reshape((h * w,), order='F')
                nbh = np.vstack((self.image[i, 0, 0:1, :], self.image[i, 0, 0:h - 1, :]))
                utmp[:, 3] = nbh.reshape((h * w,), order='F')
                nbh = np.vstack((self.image[i, 0, 1:h, :], self.image[i, 0, h - 1:h, :]))
                utmp[:, 5] = nbh.reshape((h * w,), order='F')

                Index0 = np.linspace(0,h * w, h * w, endpoint = False, dtype = int)
                dipi = dip[i]
                for d in range(7):
                    dipi[2*d:2*d + 2, :] = np.where(dipi[2*d:2*d + 2, :] < - d, - 2*d, dipi[2*d:2*d + 2, :])
                    dipi[h - 2*d - 2:h - 2*d, :] = np.where(dipi[h - 2*d - 2:h - 2*d, :] > d, 2*d, dipi[h - 2*d - 2:h - 2*d, :])
                Indexf1 = Index0 + dipi.flatten(order='F') + h
                Indexf1[h * w - h:h * w] = Index0[h * w - h:h * w]
                utmp[:, 7] = utmp[:, 2][Indexf1]
                utmp[:, 6] = utmp[:, 1][Indexf1]
                utmp[:, 8] = utmp[:, 3][Indexf1]
                Indexb1 = np.hstack((Index0[0:dipi.shape[0],], Index0[0:Index0.shape[0] - dipi.shape[0],])) # 初始化Indexb1
                Indexb1[np.flipud(Indexf1)] = np.flipud(Index0)
                utmp[:, 1] = utmp[:, 2][Indexb1]
                utmp[:, 0] = utmp[:, 3][Indexb1]
                utmp[:, 2] = utmp[:, 5][Indexb1]
                # filter
                u = np.median(utmp, axis = 1)
                somf[i, 0] = u.reshape(h, w, order='F')
            if np.sum(act[i] == self.move_range + 15) > 0:
                weight_ssim = get_ssim(self.ori[i, 0] - self.image[i, 0], self.image[i, 0])
                enhance[i, 0] = (1 + 0.1 * weight_ssim) * self.image[i, 0]


        self.image = somf
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range-3, moved_image, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range-2, box, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range-1, box0, self.image)
        self.image = np.where(act[:, np.newaxis, :, :] == self.move_range + 15, enhance, self.image)

        self.tensor[:, :self.image.shape[1], :, :] = self.image
        self.tensor[:, -64:, :, :] = inner_state
