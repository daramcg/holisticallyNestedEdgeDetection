import cv2
img = cv2.imread("QGIS_TCI_summer.tif")
(H, W) = img.shape[:2]
blob = cv2.dnn.blobFromImage(img, scalefactor = 0.1, size = (W, H), swapRB = True, crop = False)
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "hed_pretrained_bsds.caffemodel")
net.setInput(blob)
hed = net.forward()
hed = cv2.resize(hed[0,0], (W, H))
hed = (255 * hed).astype("uint8")

cv2.imwrite("hed_result_summer_tci_0.1.tif", hed)

cv2.imshow("Input", img)
cv2.imshow("HED", hed)
cv2.waitKey(0)