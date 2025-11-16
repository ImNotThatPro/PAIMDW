import cv2

cap = cv2.VideoCapture(0)

ret, prev_frame =cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print('Failed to grab frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frame_diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    trail = None

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)

        mask_roi = thresh[y:y+h, x:x+w]
        motion_level = cv2.mean(mask_roi)[0]
        
        if motion_level >180:
            color = (0, 0, 255)
        elif motion_level > 80:
            color = (0, 255, 255)
        else:
            color = (0,255,0)

        thickness = max(1, min(10, int(cv2.contourArea(contour)/ 2000)))



        overlay = frame.copy()
        alpha = 0.3

        cv2.rectangle(frame, (x,y), (x + w, y + h), color, thickness)
        cv2.addWeighted(overlay, alpha, frame, 1- alpha, 0, frame)
        
    if trail is None:
        trail = thresh.copy().astype('float')
    else:
        cv2.accumulateProduct(thresh, trail, 0.1)

    trail_display = cv2.convertScaleAbs(trail)
    combined = cv2.hconcat([frame, cv2.cvtColor(trail_display, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('Motion Detection', combined)

    prev_gray = gray.copy()

    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
