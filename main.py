import cv2
import math

cap = cv2.VideoCapture(0)

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("Trail", cv2.WINDOW_NORMAL)
cv2.namedWindow("Heatmap", cv2.WINDOW_NORMAL)
cv2.namedWindow("Combined", cv2.WINDOW_NORMAL)
cv2.namedWindow("Cinematic", cv2.WINDOW_NORMAL)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

trail = None
prev_centroid = None
heatmap = None

trackers = {}
next_id = 0

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

    if heatmap is None:
        heatmap = thresh.copy().astype('float')
    else:
        cv2.accumulateWeighted(thresh, heatmap, 0.05)

    heatmap_display = cv2.convertScaleAbs(heatmap)
    colored_heatmap = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)

    fastest_speed = -1
    fastest_box = None

    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        mask_roi = thresh[y:y + h, x:x + w]
        motion_level = cv2.mean(mask_roi)[0]

        # color logic
        if motion_level > 180:
            color = (0, 0, 255)
        elif motion_level > 80:
            color = (0, 255, 255)
        else:
            color = (0, 255, 0)

        thickness = max(1, min(10, int(cv2.contourArea(contour) / 2000)))

        overlay = frame.copy()

        cx = x + w // 2
        cy = y + h // 2
        centroid = (cx,cy)

        assigned_id = None
        min_dist = 999999

        for obj_id, data in trackers.items():
            px, py = data['prev_centroid']
            dist = math.hypot(cx-px,cy-py)

            if dist < min_dist and dist<50:
                min_dist=dist
                assigned_id = obj_id
        if assigned_id is None:
            assigned_id = next_id
            trackers[assigned_id] = {
                'prev_centroid': centroid,
                'color': (int(math.sin(next_id)* 127 + 128),
                        int(math.cos(next_id)*127 + 128),
                        200),
                'speed': 0
            }
            next_id +=1

        px,py = trackers[assigned_id]['prev_centroid']
        dx = cx-px
        dy = cy-py
        speed = math.sqrt(dx*dx + dy*dy)
        trackers[assigned_id]['speed'] = speed
        trackers[assigned_id]['prev_centroid'] = centroid

        if min_dist< 50:
            cv2.arrowedLine(frame,(px,py),(cx,cy),trackers[assigned_id]['color'],2)
        cv2.putText(frame, f'ID{assigned_id} | {speed:.1f}',
                    (x,y -10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,
                    trackers[assigned_id]['color'],2
                    )


        # FIXED fastest speed selection
        if speed > fastest_speed:
            fastest_speed = speed
            fastest_box = (x, y, w, h)



        cv2.putText(frame, f"Speed: {speed:.1f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # draw fastest box
    if fastest_box is not None:
        fx, fy, fw, fh = fastest_box
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 3)
        cv2.putText(frame, 'Fastest', (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if trail is None:
        trail = thresh.copy().astype('float')
    else:
        cv2.accumulateWeighted(thresh, trail, 0.1)

    trail_display = cv2.convertScaleAbs(trail)
    trail_bgr = cv2.cvtColor(trail_display, cv2.COLOR_GRAY2BGR)

    final_display = cv2.addWeighted(colored_heatmap, 0.4,frame, 0.6,0)
    cv2.imshow("Tracking", final_display)
    cv2.imshow("Trail", trail_bgr)
    cv2.imshow("Heatmap", colored_heatmap)
    cv2.imshow("Cinematic", final_display)

    combined = cv2.hconcat([frame, trail_bgr, colored_heatmap])
    cv2.imshow("Combined", combined)

    prev_gray = gray.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()