# import libraries
from ultralytics import YOLO
import cv2

# It downloads the model automatically. If it is not already downloaded.
# I choose the large model you can choose others. Such as, YOLOv8n-seg, YOLOv8s-seg etc.
model = YOLO("yolov8l-seg.pt")


# Automatic segmentation image
def automatic_segmentation_img(img_pth):
    # Open the img
    img = cv2.imread(img_pth)

    # Give img to the model
    result = model(img)

    # Take masked results
    annotated_img = result[0].plot()

    # Save masked image
    cv2.imwrite("../datas/test_img_result_automatic.jpg", annotated_img)


# Customized segmentation image
def customized_segmentation_img(img_pth):
    # Open the img
    img = cv2.imread(img_pth)

    # Describe image size
    w = img.shape[1]
    h = img.shape[0]

    if w == h:
        imgsz = w
    elif w > h:
        imgsz = w
    else:
        imgsz = h

    # Give img to the model
    result = model(img, imgsz=imgsz)

    # Get masks
    masks = (result[0].masks.data.cpu().numpy() * 255).astype('uint8')

    # Get boxes
    boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

    # Get indexes of object names
    clss = result[0].boxes.cls

    for box, cls, mask in zip(boxes, clss, masks):
        # Get names of objects
        name = result[0].names[int(cls)]

        # Draw Rectangle
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        # Find center of object
        center_of_object = (int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1])

        # Put a dot to center of object
        cv2.circle(img, (center_of_object), 5, (0, 0, 255), cv2.FILLED)

        # Write names and center points
        cv2.putText(img, f"{name}-{center_of_object}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Resize mask
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Convert mask's color base
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        # Change color of mask
        mask[(mask == 255).all(-1)] = [0, 130, 0]

        # Overlap image and mask (img, alpha = weight of first image, mask, beta = weight of mask)
        img = cv2.addWeighted(img, 1, mask, 0.5, 0)

    # Save result
    cv2.imwrite(f"../datas/test_img_result_customized.jpg", img)


# Automatic segmentation video
def automatic_segmentation_video(video_pth):
    # Set Video
    cap = cv2.VideoCapture(video_pth)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video writer
    writer = cv2.VideoWriter('../datas/test_video_result_automatic.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # Start
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error starting camera")
            break

        try:
            # Give img to the model
            result = model(frame)

            # Get segmented frame
            frame = result[0].plot()

        except Exception as e:
            print(e)

        # Show video
        cv2.imshow("frame", frame)

        # Write frames for a video
        writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close video writer and video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# Customized segmentation video
def customized_segmentation_video(video_pth):
    # Set Video
    cap = cv2.VideoCapture(video_pth)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Set video writer
    writer = cv2.VideoWriter('../datas/test_video_result_customized.mp4',
                             cv2.VideoWriter_fourcc(*'DIVX'), 25, (1280, 720))

    # Start
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error starting camera")
            break

        try:
            # Describe image size
            w = frame.shape[1]
            h = frame.shape[0]

            if w == h:
                imgsz = w
            elif w > h:
                imgsz = w
            else:
                imgsz = h

            # Give img to the model
            result = model(frame, imgsz=imgsz)

            # Get boxes
            boxes = result[0].boxes.xyxy.cpu().numpy().astype(int)

            # Get masks
            masks = (result[0].masks.data.cpu().numpy() * 255).astype('uint8')

            # Get indexes of object names
            clss = result[0].boxes.cls

            for box, cls, mask in zip(boxes, clss, masks):
                # Get name of object
                name = result[0].names[int(cls)]

                # Draw Rectangle
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                # Find center of object
                center_of_object = (int((box[2] - box[0]) / 2) + box[0], int((box[3] - box[1]) / 2) + box[1])

                # Put a dot to center of object
                cv2.circle(frame, (center_of_object), 5, (0, 0, 255), cv2.FILLED)

                # Write names and center points
                cv2.putText(frame, f"{name}-{center_of_object}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

                # Resize mask
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Convert mask's color base
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

                # Change color of mask
                mask[(mask == 255).all(-1)] = [0, 130, 0]

                # Overlap image and mask (img, alpha = weight of first image, mask, beta = weight of mask)
                frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)

        except Exception as e:
            print(e)

        # Show video
        cv2.imshow("frame", frame)

        # Write frames for a video
        writer.write(frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Close video writer and video
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


img_pth = "../datas/test_img.jpg"

automatic_segmentation_img(img_pth)
customized_segmentation_img(img_pth)

video_pth = "../datas/test_video.mp4"

customized_segmentation_video(video_pth)
automatic_segmentation_video(video_pth)