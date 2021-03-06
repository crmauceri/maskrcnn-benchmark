from maskrcnn_benchmark.config import cfg
from predictor import SUNSpotDemo
from cv2 import imread, imshow, waitKey, destroyAllWindows

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

sunspot_demo = SUNSpotDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)
# load image and then run prediction
image = imread("6817.jpg")
refexp = "The flowers on the table"  # Referring expression
predictions = sunspot_demo.run_on_opencv_image(image)
imshow("Detections", predictions)
waitKey(0)
destroyAllWindows()