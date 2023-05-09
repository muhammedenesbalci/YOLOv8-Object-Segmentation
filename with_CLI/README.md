## YOLOv8 object segmentation with CLI(Command Line Interface)

**Usage**

Pip install the ultralytics package including all requirements.  
  

    pip install ultralytics  

  
  
Run this command in CLI.  
  

    yolo segment predict model=yolov8l-seg.pt source ="test_video.mp4" show=true
  

**Models**

| Model                                                                                        | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>mask<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| -------------------------------------------------------------------------------------------- | --------------------- | -------------------- | --------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt) | 640                   | 36.7                 | 30.5                  | 96.1                           | 1.21                                | 3.4                | 12.6              |
| [YOLOv8s-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt) | 640                   | 44.6                 | 36.8                  | 155.7                          | 1.47                                | 11.8               | 42.6              |
| [YOLOv8m-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt) | 640                   | 49.9                 | 40.8                  | 317.0                          | 2.18                                | 27.3               | 110.2             |
| [YOLOv8l-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt) | 640                   | 52.3                 | 42.6                  | 572.4                          | 2.79                                | 46.0               | 220.5             |
| [YOLOv8x-seg](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt) | 640                   | 53.4                 | 43.4                  | 712.1                          | 4.02                                | 71.8               | 344.1             |


**Arguments**

- source	'ultralytics/assets'	source directory for images or videos
- conf	0.25	object confidence threshold for detection
- iou	0.7	intersection over union (IoU) threshold for NMS
- half	False	use half precision (FP16)
- device	None	device to run on, i.e. cuda device=0/1/2/3 or device=cpu
- show	False	show results if possible
- save	False	save images with results
- save_txt	False	save results as .txt file
- save_conf	False	save results with confidence scores
- save_crop	False	save cropped images with results
- hide_labels	False	hide labels
- hide_conf	False	hide confidence scores
- max_det	300	maximum number of detections per image
- vid_stride	False	video frame-rate stride
- line_thickness	3	bounding box thickness (pixels)
- visualize	False	visualize model features
- augment	False	apply image augmentation to prediction sources
- agnostic_nms	False	class-agnostic NMS
- retina_masks	False	use high-resolution segmentation masks
- classes	None	filter results by class, i.e. class=0, or class=[0,2,3]
- boxes True Show boxes in segmentation predictions
