import numpy as np
import math

def get_distance(p1, p2):
    """두 점 사이의 유클리드 거리"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def calculate_vertical_angle(p1, p2):
    """수직선과 두 점을 잇는 선 사이의 각도 (SR1용)"""
    dx, dy = p2.x - p1.x, p2.y - p1.y
    len_v = np.sqrt(dx*dx + dy*dy)
    if len_v == 0: return 0
    uy = dy/len_v
    angle = math.degrees(math.acos(abs(uy)))
    return angle

def calculate_lines_angle(line1_p1, line1_p2, line2_p1, line2_p2):
    """두 선분 사이의 각도 (SR2용)"""
    v1 = np.array([line1_p2.x - line1_p1.x, line1_p2.y - line1_p1.y])
    v2 = np.array([line2_p2.x - line2_p1.x, line2_p2.y - line2_p1.y])
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0
    dot_product = np.dot(v1, v2)
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    return abs(math.degrees(math.acos(cos_angle)))

def get_score_by_threshold(value, metric_type):
    """점수 환산 함수"""
    score = 0
    # SR1, SR2: 각도 기준
    if metric_type in ['SR1', 'SR2']:
        if value < 1: score = 10
        elif value < 1: score = 9
        elif value < 2: score = 8
        elif value < 3: score = 7 # 2~3도
        elif value < 4: score = 6
        elif value < 5: score = 5
        elif value < 6: score = 4
        elif value < 7: score = 3
        elif value < 8: score = 2
        else: score = 1
    # SR3 ~ SR7: 대칭 비율 기준 (유사도 %)
    else: 
        diff = abs(value - 100) # 100%와의 차이
        if diff == 0: score = 10
        elif diff <= 4: score = 9   # 96~104
        elif diff <= 29: score = 8  # 71~129
        elif diff <= 69: score = 6  # 31~169
        elif diff <= 94: score = 3  # 6~194
        elif diff <= 99: score = 2  # 1~199
        else: score = 1             # 0 or >199
    return score

def get_face_features(landmarks):
    p = lambda i: landmarks[i]
    feats = {}

    # 1. SR1용 각도 (미간-윗입술 수직각)
    feats['vertical_angle'] = calculate_vertical_angle(p(168), p(0))

    # 2. SR2용 각도 (눈꼬리선 vs 입꼬리선)
    feats['parallel_angle'] = calculate_lines_angle(p(33), p(263), p(61), p(291))

    # 3. 눈썹 끝점 - 앞광대 거리 (SR3, SR4, SR5 공통 측정 요소)
    # 좌측(46-123), 우측(276-352)
    l_brow_cheek = get_distance(p(46), p(123))
    r_brow_cheek = get_distance(p(276), p(352))
    if r_brow_cheek == 0: feats['brow_cheek_ratio'] = 0
    else: feats['brow_cheek_ratio'] = (l_brow_cheek / r_brow_cheek) * 100

    # 4. 눈꼬리 - 입꼬리 거리 (SR6, SR7 공통 측정 요소)
    # 좌측(33-61), 우측(263-291)
    l_eye_mouth = get_distance(p(33), p(61))
    r_eye_mouth = get_distance(p(263), p(291))
    if r_eye_mouth == 0: feats['eye_mouth_ratio'] = 0
    else: feats['eye_mouth_ratio'] = (l_eye_mouth / r_eye_mouth) * 100

    return feats

def calculate_final_report(steps_data):
    """
    4단계 데이터를 모두 모아서 최종 점수(RI) 계산
    steps_data: [무표정feats, 눈감기feats, 이마feats, 미소feats]
    """
    # 데이터 매핑
    resting = steps_data[0] # 무표정
    closed  = steps_data[1] # 눈감기
    raised  = steps_data[2] # 이마
    smile   = steps_data[3] # 미소

    scores = {}
    raw_vals = {}

    # --- 개별 지표 계산 ---
    # SR1 (무표정)
    scores['SR1'] = get_score_by_threshold(resting['vertical_angle'], 'SR1')
    raw_vals['SR1'] = resting['vertical_angle']

    # SR2 (무표정)
    scores['SR2'] = get_score_by_threshold(resting['parallel_angle'], 'SR2')
    raw_vals['SR2'] = resting['parallel_angle']

    # SR3 (무표정 - 눈썹광대)
    scores['SR3'] = get_score_by_threshold(resting['brow_cheek_ratio'], 'SR3')
    raw_vals['SR3'] = resting['brow_cheek_ratio']

    # SR4 (눈감기 - 눈썹광대)
    scores['SR4'] = get_score_by_threshold(closed['brow_cheek_ratio'], 'SR4') # 타입은 SR3과 같음
    raw_vals['SR4'] = closed['brow_cheek_ratio']

    # SR5 (이마 - 눈썹광대)
    scores['SR5'] = get_score_by_threshold(raised['brow_cheek_ratio'], 'SR5')
    raw_vals['SR5'] = raised['brow_cheek_ratio']

    # SR6 (무표정 - 눈입거리)
    scores['SR6'] = get_score_by_threshold(resting['eye_mouth_ratio'], 'SR6')
    raw_vals['SR6'] = resting['eye_mouth_ratio']

    # SR7 (미소 - 눈입거리)
    scores['SR7'] = get_score_by_threshold(smile['eye_mouth_ratio'], 'SR7') # 타입은 SR6과 같음
    raw_vals['SR7'] = smile['eye_mouth_ratio']

    # --- RI (Risk Index) 종합 점수 계산 ---
    # 공식: (SR1 + SR2 + Avg(SR3~SR7)) / 30 * 100
    avg_symmetry = (scores['SR3'] + scores['SR4'] + scores['SR5'] + scores['SR6'] + scores['SR7']) / 5
    
    ri_score = (scores['SR1'] + scores['SR2'] + avg_symmetry) / 30 * 100
    
    scores['RI_Score'] = round(ri_score, 1)

    # 등급 판정
    if ri_score >= 90: grade = "(Excellent)"
    elif ri_score >= 70: grade = "(Good)"
    elif ri_score >= 50: grade = "(Fair)"
    elif ri_score >= 30: grade = "(Poor)"
    else: grade = "(Severe)"
    
    scores['RI_Grade'] = grade

    return scores, raw_vals