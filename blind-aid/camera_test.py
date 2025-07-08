import cv2

# Try different indexes: 0, 1, 2...
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found!!")
else:
    print("Camera opened successfully.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed.")
            break
        cv2.imshow("Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
