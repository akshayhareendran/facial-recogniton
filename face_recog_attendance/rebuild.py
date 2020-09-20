import cv2
import numpy as np
import face_recognition
import os

path = 'faces'
images = []
student_names = []
myList = os.listdir(path)       #
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    student_names.append(os.path.splitext(cl)[0])
print(student_names) # just the names without extensions

test_img = face_recognition.load_image_file('group.jpg')
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
img = test_img


def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #print(encode)
    return encodeList

main_encode = find_encodings(images)

#
# main_img = face_recognition.load_image_file('images/Arju.jpg')
# main_img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
# face_loc = face_recognition.face_locations(main_img)[0]
# main_encode = face_recognition.face_encodings(main_img)[0]
# cv2.rectangle(main_img, (face_loc[3], face_loc[0]), (face_loc[1], face_loc[2]), (255, 255, 0), 2)




face_loc_test = face_recognition.face_locations(test_img)
encodes_current_frame = face_recognition.face_encodings(test_img, face_loc_test)

#print(face_loc_test,encodes_current_frame)
#for i in range(len(face_loc_test)):

for i in range(len(face_loc_test)):
        encoding_to_check = encodes_current_frame[i]
        cv2.rectangle(test_img,(face_loc_test[i][1] - 10,face_loc_test[i][0] + 10),(face_loc_test[i][3]+10,face_loc_test[i][2]+10),(255,0,255),2)
        matches = face_recognition.compare_faces(main_encode, encoding_to_check,0.3)
        face_distance = face_recognition.face_distance(main_encode,encoding_to_check)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            screen= student_names[match_index]
            y1, x2, y2, x1 = face_loc_test[match_index]
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,screen, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

#cv2.imshow('MAIN', main_img)
cv2.imshow('test', test_img)

cv2.waitKey(0)

cv2.destroyAllWindows()