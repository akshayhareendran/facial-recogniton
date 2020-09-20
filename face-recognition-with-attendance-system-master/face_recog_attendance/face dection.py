import cv2
import face_recognition

image = face_recognition.load_image_file("images/c634fd63-b564-4187-8877-56536b227320.jpg")
face_loc_test = face_recognition.face_locations(image)
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

print(face_loc_test)
print("[INFO] Found {0} Faces.".format(len(face_loc_test)))
for i in range(len(face_loc_test)):
    image=cv2.rectangle(image,(face_loc_test[i][1],face_loc_test[i][0]),(face_loc_test[i][3],face_loc_test[i][2]),(255,0,255),2)

path= 'faces'

for i in range(len(face_loc_test)):
    image=cv2.rectangle(image,(face_loc_test[i][1],face_loc_test[i][0]),(face_loc_test[i][3],face_loc_test[i][2]),(255,0,255),2)
    x, y, w, h = face_loc_test[i]
    roi_color =image[x:w,h:y]
    print("[INFO] Object found. Saving locally.")
    #cv2.imwrite(str(i + 1000) + '_faces.jpg',roi_color)
    cv2.imwrite(f'{path}/{i}.jpg',roi_color)

status = cv2.imwrite('faces_detected.jpg', image)
print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


cv2.imshow('Output',image)
cv2.waitKey(0)