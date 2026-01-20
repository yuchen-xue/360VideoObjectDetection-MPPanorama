from ultralytics import YOLO


def detect(
    yolo_model,
    input_array,
    confidence_threshold=0.40,
):
    """
    Function:
    Load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Input:
        yolo_model (str): Path to the YOLO model.
        input_array (np.ndarray): Path to the input image.

    Output:
        detections: Array of detections (list) with info such as class_id, class_name, confidence, etc.
    """
    # Load the YOLO model
    model = YOLO(yolo_model)

    # Margin to filter boxes close to image border
    margin = 0.02

    # Perform inference
    results = model.predict(
        source=input_array, conf=confidence_threshold, save=False, verbose=False
    )

    detections: list[dict] = []

    for result in results:
        # Calculate scale factor of the input image
        length = max(result.orig_shape)
        scale = length / 640

        # Iterate through detection results
        for j in range(len(result.boxes)):

            # Get each bounding box
            box = result.boxes[j]

            # Get normalized box coordinates
            x_norm, y_norm, w_norm, h_norm = box.xywhn[0]

            left_norm = x_norm - w_norm / 2
            right_norm = x_norm + w_norm / 2
            top_norm = y_norm - h_norm / 2
            bottom_norm = y_norm + h_norm / 2

            # Skip boxes too close to the image border
            if (
                left_norm <= margin
                or top_norm <= margin
                or right_norm >= 1 - margin
                or bottom_norm >= 1 - margin
            ):
                continue

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
