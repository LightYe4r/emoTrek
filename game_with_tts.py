import cv2 as cv 
import numpy as np
import mediapipe as mp
import math
import serial
import time
import vlc

#ser=serial.Serial('com7',9600)

mp_face_mesh = mp.solutions.face_mesh

flag = True

write_flag = True

music_flag = True

NOSE_LANDMARK = 1

dest_x, dest_y, dest_y2 = 200, 100, 300
dest_x, dest_y, dest_y2 = 200, 100, 300
Bound= 30
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

    player_generator("./music/audio_0_게임을_시작해볼까요_.mp3").play()
    time.sleep(1)
    start = time.time()
    while True:
        check = True
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        results = face_mesh.process(rgb_frame)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            
            # 얼굴 감지
            faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.2, minNeighbors=1, minSize=(50, 50))

            # 각 얼굴에 대해 미소 감지 수행
            for (x, y, w, h) in faces:
                roi_gray = rgb_frame[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # 미소 감지 수행
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=22, minSize=(40, 40))

                # 얼굴에 미소가 감지된 경우에만 사각형 표시
                if len(smiles) > 0:
                    for (sx, sy, sw, sh) in smiles:
                        cv.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                        smile_count+=2
                        if(smile_count > 10):
                            draw_text(frame,"check smile",100,140,red)
                            smile_count = 0
                            result += 1
                        
                else:
                    smile_count -= 1
                    if smile_count < 0:
                        smile_count = 0
                    #draw_text(frame,"check !smile",100,140,red)
                    
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


            dest_y2 = 200+dest_y
            if dest_y2 > 300:
                dest_y2 = 300
            elif dest_y2 < 150:
                dest_y2 = 150    

            dest = str(int(dest_x)) +" " + str(int(dest_y)) +" "+ str(int(dest_y2))
            #print(dest, sumx, sumy)
            #ser.write(dest.encode('utf-8'))                         

        curr_time = time.time()
        time_passed = curr_time - start

        if(check):
            text_color = green
        else:
            text_color = red

        if check:
            if(start_flag):
                draw_text(frame,"IN",480,40,text_color)
            if(time_passed < 13):
                frame = cv.rectangle(frame,(310,250),(330,230),(0,255,0),3)
                if(start_flag):
                    count = count + 1
        else:
            if(start_flag):
                draw_text(frame,"OUT",480,40,text_color)
            if(time_passed < 13):
                frame = cv.rectangle(frame,(310,250),(330,230),(0,0,255),3)


        if(time_passed < 1):
            draw_text(frame,"3",310,230,text_color)
        elif(time_passed < 2):
            draw_text(frame,"2",310,230,text_color)
        elif(time_passed<3):
            draw_text(frame,"1",310,230,text_color)
        else:
            start_flag = True

        if(time_passed < 4 and start_flag):
            draw_text(frame,"10",480,100,text_color)
        elif(time_passed < 5 and start_flag):
            draw_text(frame,"9",480,100,text_color)
        elif(time_passed < 6 and start_flag):
            draw_text(frame,"8",480,100,text_color)
        elif(time_passed < 7 and start_flag):
            draw_text(frame,"7",480,100,text_color)
        elif(time_passed < 8 and start_flag):
            draw_text(frame,"6",480,100,text_color)
        elif(time_passed < 9 and start_flag):
            draw_text(frame,"5",480,100,text_color)
        elif(time_passed < 10 and start_flag):
            if(music_flag):
                music_flag = False
                player_generator("./music/audio_1_잘하고_있습니다.mp3").play()
            draw_text(frame,"4",480,100,text_color)
        elif(time_passed < 11 and start_flag):
            draw_text(frame,"3",480,100,text_color)
        elif(time_passed < 12 and start_flag):
            draw_text(frame,"2",480,100,text_color)
        elif(time_passed < 13 and start_flag):
            draw_text(frame,"1",480,100,text_color)
        elif(time_passed >= 13 and start_flag):
            stayed_time = count/30
            stayed_time = round(stayed_time,3)
            draw_text(frame,"you stayed " + str(stayed_time)+ " seconds" + str(result) + "smiles",100,140,red)
        
            if(write_flag):
                if(stayed_time >= 5):
                    player_generator("./music/audio_2_굉장한데요_.mp3").play()
                else:
                    player_generator("./music/audio_3_아쉽지만_다음_기회에___.mp3").play()
                write_flag = False
                with open("result.txt", "a") as file:
                    file.write(str(stayed_time) + "\n")
                    file.close()

                with open("result.txt", "r") as file:
                    lines = [float(line.strip()) for line in file]

                    lines.sort(reverse=True)

                with open("result.txt", "w") as file:
                    for value in lines:
                        file.write(str(value) + "\n")

            with open("result.txt", "r") as file:
                lines = [float(line.strip()) for line in file]
                ranking = 1
                for values in lines[:5]:
                    draw_text(frame,str(ranking) +". " + str(values)+ " seconds",100,160 + 40 * ranking,red)
                    ranking = ranking + 1


        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break

cap.release()
cv.destroyAllWindows()