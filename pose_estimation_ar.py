import cv2 as cv
import numpy as np

# 🎯 캘리브레이션 결과 불러오기
with np.load('calibration_params.npz') as data:
    K = data['camera_matrix']
    dist_coeff = data['dist_coeffs']

# 🎯 체스보드 설정
board_pattern = (7, 6)          # 내부 코너 수
board_cellsize = 1.0            # 칸 크기 (단위: cm 또는 임의의 단위)

# 체스보드 3D 포인트 생성 (Z=0 평면에 있음)
obj_points = np.array([[c, r, 0] for r in range(board_pattern[1])
                                  for c in range(board_pattern[0])], dtype=np.float32) * board_cellsize

# 3D AR 박스 만들기 (윗면은 Z = -1, 아래는 Z = 0)
box_lower = board_cellsize * np.array([[4, 2, 0], [5, 2, 0], [5, 4, 0], [4, 4, 0]], dtype=np.float32)
box_upper = board_cellsize * np.array([[4, 2, -1], [5, 2, -1], [5, 4, -1], [4, 4, -1]], dtype=np.float32)

# 체스보드 코너 정밀화 조건
board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 🎥 동영상 열기 (sample_video.mp4는 같은 폴더에 있어야 함)
cap = cv.VideoCapture('chessboard.mp4')

if not cap.isOpened():
    print("❌ 동영상 파일을 열 수 없습니다.")
    exit()

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break  # 영상 끝

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ✅ 체스보드 인식
    complete, img_points = cv.findChessboardCorners(gray, board_pattern)
    
    if complete:
        img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1), board_criteria)

        # ✅ 카메라 포즈 추정
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # ✅ 3D 박스 투영 및 그리기
        line_lower, _ = cv.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
        line_upper, _ = cv.projectPoints(box_upper, rvec, tvec, K, dist_coeff)

        line_lower = np.int32(line_lower).reshape(-1, 2)
        line_upper = np.int32(line_upper).reshape(-1, 2)

        # 사각형(빨강, 파랑)과 연결선(초록)
        cv.polylines(img, [line_lower], True, (0, 0, 255), 2)
        cv.polylines(img, [line_upper], True, (255, 0, 0), 2)
        for b, t in zip(line_lower, line_upper):
            cv.line(img, tuple(b), tuple(t), (0, 255, 0), 2)

        # 카메라 위치 계산 및 출력
        R, _ = cv.Rodrigues(rvec)
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 체스보드 시각화
        cv.drawChessboardCorners(img, board_pattern, img_points, complete)

    cv.imshow('Camera Pose Estimation from Video', img)

    if cv.waitKey(30) == 27:  # ESC 누르면 종료
        break

cap.release()
cv.destroyAllWindows()
