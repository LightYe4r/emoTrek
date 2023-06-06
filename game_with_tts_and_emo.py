import cv2 as cv 
import numpy as np
import mediapipe as mp
import math
import serial
import time
import vlc

#ser=serial.Serial('com6',9600)

mp_face_mesh = mp.solutions.face_mesh

title_image = cv.imread("./title.png",cv.IMREAD_COLOR)
description_image = cv.imread("./descript.png",cv.IMREAD_COLOR)
title_flag = True
description_flag = False
game_flag = False

box_length = 50

flag = True

write_flag = True

music_flag = True

NOSE_LANDMARK = 1

dest_x, dest_y, dest_y2 = 200, 100, 300
Bound= 50
Ki=0.001
Kp=40

dest = "90,45,135"

cap = cv.VideoCapture(0)

blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255) 

face_cascade_path = 'haarcascade_frontalface_default.xml'
smile_cascade_path = 'haarcascade_smile.xml'

face_cascade = cv.CascadeClassifier(face_cascade_path)
smile_cascade = cv.CascadeClassifier(smile_cascade_path)

smile_count = 0
result = 0
miso_check = False
def player_generator(filename):
    instance = vlc.Instance()

    player = instance.media_player_new()

    media = instance.media_new(filename)

    player.set_media(media)

    return player

def draw_text(img, text, x, y,text_color):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_color=text_color

    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    offset = 5

    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    
def convert(x, max, min, dest_max, dest_min):
    converted_x = dest_min + (x-min) * ((dest_max-dest_min) / (max - min))
    return converted_x

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    start_flag = False

    count = 0

    sumx = 0
    sumy = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        if results.multi_face_landmarks:
            #각도보정 시작부근
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                  for p in results.multi_face_landmarks[0].landmark])
            angle1_x, angle1_y = mesh_points[94]
            angle2_x, angle2_y = mesh_points[197]
            if(angle2_y-angle1_y!=0):
                angle=(angle2_x-angle1_x)/(angle2_y-angle1_y)
                angle= np.arctan(angle)
                angle= np.rad2deg(angle)
            else:
                if(angle2_x-angle1_x<0):
                    angle=90
                else:
                    angle=-90
            rotation_matrix = cv.getRotationMatrix2D((img_w / 2, img_h / 2), -angle, 1)
            rotated_frame = cv.warpAffine(rgb_frame, rotation_matrix, (img_w, img_h))
            #각도 보정끝->각도 보정된 이미지는 rotated_frame으로 저장->rotated frame으로 얼굴 감지
            # 얼굴 감지
            faces = face_cascade.detectMultiScale(rotated_frame, scaleFactor=1.2, minNeighbors=1, minSize=(0, 0))
            # 각 얼굴에 대해 미소 감지 수행
            for (x, y, w, h) in faces:
                roi_gray = rotated_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

            # 미소 감지 수행
            smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=22, minSize=(0, 0))
            # 얼굴에 미소가 감지된 경우에만 사각형 표시
            if len(smiles) > 0:
                for (sx, sy, sw, sh) in smiles:
                    print("check for miso")
                    cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                    smile_count+=1
                    if(smile_count > 15):
                        smile_count = 0
                        if title_flag:
                            cv.imshow("title",title_image)
                            title_flag = False
                            description_flag = True
                        elif description_flag:
                            cv.imshow("title",description_image)
                            description_flag = False
                            game_flag = True
                            player_generator("./music/audio_0_게임을_시작해볼까요_.mp3").play()
                            time.sleep(1)
                            start = time.time()
                        elif game_flag and miso_check:
                            result += 1
                            miso_check = False

            else:
                smile_count -= 1
                if smile_count < 0:
                    smile_count = 0
        if title_flag:
            cv.imshow("title",title_image)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            if key == ord('p'):
                description_flag = True
                title_flag = False
        if description_flag:
            cv.imshow("title",description_image)
            key = cv.waitKey(1)
            if key ==ord('q'):
                break
            if key == ord('p'):
                description_flag = False
                game_flag = True
                player_generator("./music/audio_0_게임을_시작해볼까요_.mp3").play()
                time.sleep(1)
                start = time.time()
        if game_flag:
            check = True
            if results.multi_face_landmarks:
                
                mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                for p in results.multi_face_landmarks[0].landmark])
                
                x, y = mesh_points[1]
                left_nose_x, left_nose_y = mesh_points[331]
                right_nose_x, right_nose_y = mesh_points[102]

                x = x - img_w/2
                y = -y + img_h/2
                left_nose_x = left_nose_x - img_w/2
                left_nose_y = -left_nose_y + img_h/2
                right_nose_x = right_nose_x - img_w/2
                right_nose_y = -right_nose_y + img_h/2
                                
                if(x > Bound and dest_x < 380):
                    if(sumx < 10000):
                        sumx += x
                    elif(sumx >= 10000):
                        sumx = 10000
                    dest_x=dest_x+int(x/Kp)+sumx*Ki
                    check = False
                elif(x< -Bound and dest_x > 20):
                    if(sumx > -10000):
                        sumx += x
                    elif(sumx <=-10000):
                        sumx = -10000
                    dest_x=dest_x+int(x/Kp)+sumx*Ki
                    check = False

                if(y> Bound and dest_y < 380):
                    if(sumy < 10000):
                        sumy += y
                    elif(sumy>=10000):
                        sumy = 10000
                    dest_y=dest_y+int(y/Kp)+sumy*Ki
                    check = False
                elif(y < -Bound and dest_y > 20):
                    if(sumy > -10000):
                        sumy += y
                    elif(sumy<=-10000):
                        sumy = -10000
                    dest_y=dest_y+int(y/Kp)+sumy*Ki
                    check = False
                
                if not check:
                    miso_check = True
                dest_y2 = 200+dest_y
                if dest_y2 > 300:
                    dest_y2 = 300
                elif dest_y2 < 150:
                    dest_y2 = 150    

                dest = str(int(dest_x)) +" " + str(int(dest_y)) +" "+ str(int(dest_y2))

                #ser.write(dest.encode('utf-8'))                         

            curr_time = time.time()
            time_passed = curr_time - start

            if(miso_check):#(check):
                text_color = green
            else:
                text_color = red

            if check:
                if(start_flag):
                    if(time_passed < 18):
                        frame = cv.rectangle(frame,(320-box_length,240+box_length),(320 + box_length,240 - box_length),(0,255,0),3)
                        if(start_flag):
                            count = count + 1
            else:
                if(start_flag):
                    if(time_passed < 18):
                        frame = cv.rectangle(frame,(320-box_length,240+box_length),(320 + box_length,240 - box_length),(0,255,0),3)
                        if(x > 0):
                            frame = cv.line(frame, (320 + box_length, 240+box_length), (320 + box_length, 240-box_length), red,5)
                        else:
                            frame = cv.line(frame, (320 - box_length, 240+box_length), (320 - box_length, 240-box_length), red,5)
                        if(y > 0):
                            frame = cv.line(frame, (320 + box_length, 240-box_length), (320 - box_length, 240-box_length), red,5)
                        else:
                            frame = cv.line(frame, (320 + box_length, 240 + box_length), (320 - box_length, 240 + box_length), red,5)

            draw_text(frame,str(result),0,0,text_color)
            if(time_passed < 1):
                draw_text(frame,"3",330,230,text_color)
            elif(time_passed < 2):
                draw_text(frame,"2",330,230,text_color)
            elif(time_passed<3):
                draw_text(frame,"1",330,230,text_color)
            else:
                start_flag = True
            if(time_passed < 4 and start_flag):
                draw_text(frame,"15",480,100,text_color)
            elif(time_passed < 5 and start_flag):
                draw_text(frame,"14",480,100,text_color)
            elif(time_passed < 6 and start_flag):
                draw_text(frame,"13",480,100,text_color)
            elif(time_passed < 7 and start_flag):
                draw_text(frame,"12",480,100,text_color)
            elif(time_passed < 8 and start_flag):
                draw_text(frame,"11",480,100,text_color)
            elif(time_passed < 9 and start_flag):
                draw_text(frame,"10",480,100,text_color)
            elif(time_passed < 10 and start_flag):
                if(music_flag):
                    music_flag = False
                    player_generator("./music/audio_1_잘하고_있습니다.mp3").play()
                draw_text(frame,"9",480,100,text_color)
            elif(time_passed < 11 and start_flag):
                draw_text(frame,"8",480,100,text_color)
            elif(time_passed < 12 and start_flag):
                draw_text(frame,"7",480,100,text_color)
            elif(time_passed < 13 and start_flag):
                draw_text(frame,"6",480,100,text_color)
            elif(time_passed < 14 and start_flag):
                draw_text(frame,"5",480,100,text_color)
            elif(time_passed < 15 and start_flag):
                draw_text(frame,"4",480,100,text_color)
            elif(time_passed < 16 and start_flag):
                draw_text(frame,"3",480,100,text_color)
            elif(time_passed < 17 and start_flag):
                draw_text(frame,"2",480,100,text_color)
            elif(time_passed < 18 and start_flag):
                draw_text(frame,"1",480,100,text_color)
            elif(time_passed >= 18 and start_flag):
                stayed_time = count/30
                stayed_time = round(stayed_time,3)
                score = str(round(15 - stayed_time, 3))
                if(result < 10):
                    draw_text(frame,"Fail!",480,100,text_color)
                else:
                    draw_text(frame,"Success!",480,100,text_color)
            cv.imshow('title', frame)
            key = cv.waitKey(1)
            if key ==ord('q'):
                break

cap.release()
cv.destroyAllWindows()