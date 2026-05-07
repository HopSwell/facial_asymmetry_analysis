import cv2
import time
import os
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
from metrics_calc import get_face_features, calculate_final_report

# MediaPipe
try:
    mp_solutions = mp.solutions
except AttributeError:
    try:
        from mediapipe.python import solutions as mp_solutions
    except ImportError:
        exit()

mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

# TTS 찾기
is_speaking_now = False # 현재 말하고 있는지 확인하는 변수

def speak_async(text):
    global is_speaking_now
    if is_speaking_now: return

    def _speak():
        global is_speaking_now
        is_speaking_now = True
        
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if "Korea" in voice.name or "KR" in voice.id:
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except:
            pass
        finally:
            is_speaking_now = False # 2002 말하기 끝남

    threading.Thread(target=_speak, daemon=True).start()

# 환경 설정
SAVE_DIR = "captured_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 가이드라인 설정
ELLIPSE_CENTER_X_RATIO = 0.5
ELLIPSE_CENTER_Y_RATIO = 0.5
ELLIPSE_AXIS_W_RATIO = 0.25
ELLIPSE_AXIS_H_RATIO = 0.35

# 모델 및 카메라초기화
mp_face_mesh = mp_solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# 시나리
steps_prompts_text = [
    "Look Forward (Resting)",
    "Close Eyes",
    "Raise Eyebrows",
    "Big Smile"
]
steps_prompts_voice = [
    "카메라를 정면으로 응시해주세요.",
    "눈을 지그시 감아주세요.",
    "눈을 크게 뜨거나 눈썹을 올려주세요.",
    "웃어주세요."
]
step_filenames = ["0_resting.jpg", "1_closed.jpg", "2_raised.jpg", "3_smile.jpg"]

current_step = 0
collected_features = []
is_counting = False
start_time = None
COUNTDOWN_TIME = 3

last_warning_time = 0
WARNING_INTERVAL = 5
step_spoken = False # 2002 현재 단계의 안내 멘트를 했는지 여부

print("시스템 시작")

#0201
speak_async("얼굴을 가이드라인에 맞춰주세요.")

# 메인 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    center_x = int(w * ELLIPSE_CENTER_X_RATIO)
    center_y = int(h * ELLIPSE_CENTER_Y_RATIO)
    axis_w = int(w * ELLIPSE_AXIS_W_RATIO)
    axis_h = int(h * ELLIPSE_AXIS_H_RATIO)

    results = face_mesh.process(rgb_frame)
    face_detected = False
    
    # 결과 처리
    if current_step >= 4:
        scores, raw_vals = calculate_final_report(collected_features)
        
        result_board = frame.copy()
        result_board = cv2.addWeighted(result_board, 0.3, np.zeros(result_board.shape, result_board.dtype), 0.7, 0)
        
        cv2.putText(result_board, "Diagnosis Complete", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(result_board, f"RI Score: {scores['RI_Score']}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result_board, f"Grade: {scores['RI_Grade']}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.putText(result_board, f"SR1: {scores['SR1']} | SR2: {scores['SR2']}", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(result_board, f"SR3:{scores['SR3']} | SR4:{scores['SR4']}", (50, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(result_board, f"SR5:{scores['SR5']} | SR7:{scores['SR7']}", (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(result_board, "Press ESC to Exit", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Patient Monitor', result_board)
        
        if not step_spoken: 
            #0201
            speak_async("모든 촬영이 끝났습니다. 결과를 확인해 주세요.")
            step_spoken = True

        print("\n최종 진단 결과 : ")
        print(f"종합 점수(RI): {scores['RI_Score']}점")
        print(f"판정 등급: {scores['RI_Grade']}")
        print("이상입니다.\n")
        
        if cv2.waitKey(0) & 0xFF == 27:
            break
        break
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            
            indices = [234, 454, 10, 152]
            points_coords = []
            for idx in indices:
                lx, ly = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                points_coords.append((lx, ly))
                cv2.circle(frame, (lx, ly), 5, (0, 0, 255), -1)

            x_vals = [p[0] for p in points_coords]
            y_vals = [p[1] for p in points_coords]
            e_left, e_right = center_x - axis_w, center_x + axis_w
            e_top, e_bottom = center_y - axis_h, center_y + axis_h
            margin = 20
            
            if (min(x_vals) > e_left - margin and max(x_vals) < e_right + margin and
                min(y_vals) > e_top - margin and max(y_vals) < e_bottom + margin):
                face_detected = True

            # 촬영 로직
            if face_detected:
                if not step_spoken:
                    speak_async(steps_prompts_voice[current_step])
                    step_spoken = True
                    is_counting = False
                    start_time = None

                if is_speaking_now:
                    cv2.putText(frame, "Listening...", (center_x-70, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    is_counting = False 
                else:
                    if not is_counting:
                        is_counting = True
                        start_time = time.time()
                    
                    elapsed = time.time() - start_time
                    remaining = COUNTDOWN_TIME - int(elapsed)

                    cv2.putText(frame, f"{remaining}", (center_x-20, center_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)

                    if remaining <= 0:
                        save_path = os.path.join(SAVE_DIR, step_filenames[current_step])
                        cv2.imwrite(save_path, frame)
                        print(f"[{current_step+1}/4단계 완료] 저장")
                        
                        feats = get_face_features(landmarks)
                        collected_features.append(feats)
                        
                        current_step += 1
                        is_counting = False
                        start_time = None
                        step_spoken = False
                        
                        cv2.putText(frame, "Next Step", (center_x-100, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                        cv2.imshow('Patient Monitor', frame)
                        cv2.waitKey(1000) 
            
            # 경고 (가이드라인 벗어남)
            else:
                is_counting = False
                start_time = None
                
                if not is_speaking_now:
                    current_time = time.time()
                    if current_time - last_warning_time > WARNING_INTERVAL:
                        speak_async("가이드라인 안으로 들어와, 정면을 응시해주세요.")
                        last_warning_time = current_time

    # 얼굴이 보이지 않을 경우
    else:
        if not is_speaking_now:
            current_time = time.time()
            if current_time - last_warning_time > WARNING_INTERVAL:
                speak_async("얼굴이 보이지 않습니다.")
                last_warning_time = current_time

    instruction = steps_prompts_text[current_step] if current_step < 4 else "Done"
    cv2.putText(frame, instruction, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    color = (0, 255, 0) if face_detected else (255, 255, 255)
    cv2.ellipse(frame, (center_x, center_y), (axis_w, axis_h), 0, 0, 360, color, 2)

    cv2.imshow('Patient Monitor', frame)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()