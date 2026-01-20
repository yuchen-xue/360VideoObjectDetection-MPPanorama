from ultralytics import YOLO


def detect(
    input_array,
    confidence_threshold=0.40,
    onnx_model="model-runs/detect/train/weights/yolov8n.onnx",
):
    """
    Function:
    Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Input:
        onnx_model (str): Path to the ONNX model.
        input_array (np.ndarray): Path to the input image.

    Output:
        detections: Array of detections (list) with info such as class_id, class_name, confidence, etc.
    """
    # Load the ONNX model
    model = YOLO(onnx_model)

    # Perform inference
    results = model.predict(source=input_array, conf=confidence_threshold, save=False, verbose=False)

    detections: list[dict] = []

    for result in results:
        # Calculate scale factor of the input image
        length = max(result.orig_shape)
        scale = length / 640

        # Iterate through detection results
        for j in range(len(result.boxes)):
            box = result.boxes[j]

            # Get x, y coordinates of the top-left corner
            box_xyxy = box.xyxy[0]
            topleft_x = int(box_xyxy[0].item())
            topleft_y = int(box_xyxy[1].item())

            # Get box width and height
            box_xywh = box.xywh[0]
            width = int(box_xywh[2].item())
            height = int(box_xywh[3].item())

            # Get class id, class name, and confidence score
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            confidence = box.conf.item()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "box": [topleft_x, topleft_y, width, height],
                    "scale": scale,
                }
            )

    return detections
