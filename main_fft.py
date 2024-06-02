import cv2, torch, os
import numpy as np
from pathlib import Path

pp = Path('./sysu')
for path in pp.rglob("*"):
	if str(path).split('.')[-1]=='jpg':
		img = cv2.imread(str(path), 0)
		# cv2.imshow('img', img)
		# cv2.waitKey(2000)
		f = np.fft.fft2(img)
		fshift = np.fft.fftshift(f)
		rows,cols = img.shape
		crow,ccol = int(rows/2),int(cols/2)
		fshift[crow-5:crow+5,ccol-5:ccol+5] = 0
		ishift = np.fft.ifftshift(fshift)
		iimg = np.fft.ifft2(ishift)
		iimg = np.abs(iimg)
		iimg = torch.from_numpy(iimg).type(torch.uint8).numpy()
		# cv2.imshow('iimg', iimg)
		# cv2.waitKey(2000)
		path_HF = path.replace('sysu', 'sysu_HF')
		os.makedirs(path_HF, exist_ok=True)
		cv2.imwrite(os.path.join(path_HF, split_[-1]), iimg)
		print(os.path.join(path_HF, split_[-1]))



