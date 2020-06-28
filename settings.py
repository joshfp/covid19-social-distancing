visual = {
    'OK_COLOR': (0, 255, 0),
    'HIGHLIGHT_COLOR': (255, 0, 0),
    'ALPHA': 50,
    'BBOX_WIDTH': 3,
    'LINE_WIDTH': 3,
    'POINT_SZ': 8
}

camera_calibration = {
    'perspective_pts': ((715, 327), (1186, 320), (1285, 738), (632, 751)),
    'distance_pts': ((1285, 738), (632, 751)),
    'distance_value': 5, # meters
    'image_size': (1920, 920),
}

detection_thresholds = {
    'distance': 1.5, # meters
    'confidence': 0.5
}

detection_trained_model_path = 'Pedestron/trained_models/epoch_34.pth.stu'
