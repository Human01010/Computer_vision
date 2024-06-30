import glob
import cv2
import numpy as np
import matplotlib as plt
# 设置棋盘格的尺寸 (内角点的数量)
chessboard_size = (8, 5)
# 设置棋盘格的每个方格的大小 (实际的测量单位，如毫米)
square_size = 16.8

# 终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备棋盘格的3D点 (例如 (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0))
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# 用于存储棋盘格的3D点和2D点的数组
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 读取图像
images = glob.glob('D:\Pycharmprojects\lessons\Computer_Vision\photo_camera/*.jpg')

# 初始化一个变量来存储灰度图像的尺寸
gray_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 保存灰度图像的尺寸
    gray_shape = gray.shape[::-1]

    # 找到棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了角点，则添加到列表中
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # 显示角点
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# 检查是否所有图像都成功读取和处理
if gray_shape is None:
    raise ValueError("未能在任何图像中找到棋盘格角点。")

# 执行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_shape, None, None)

# 输出标定结果
print("相机矩阵 (内参):")
print(camera_matrix)

print("\n畸变系数:")
print(dist_coeffs)

print("\n旋转向量 (外参):")
print(rvecs)

print("\n平移向量 (外参):")
print(tvecs)

#################################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 使用SIFT特征检测器
sift = cv2.SIFT_create()

# 函数：加载图像并进行检查
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img

# 提供图像路径
image_paths = [
    r'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/1.jpg',
    r'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/2.jpg',
    r'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/3.jpg'
]

# 加载图像并进行检查
img1 = load_image(image_paths[0])
img2 = load_image(image_paths[1])
img3 = load_image(image_paths[2])

# 检测特征点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

# 使用FLANN进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配特征点 (img1 <-> img2)
matches1_2 = flann.knnMatch(des1, des2, k=2)

# 匹配特征点 (img2 <-> img3)
matches2_3 = flann.knnMatch(des2, des3, k=2)

# 匹配特征点 (img1 <-> img3)
matches1_3 = flann.knnMatch(des1, des3, k=2)

# 使用比值测试筛选匹配
def filter_matches(matches, ratio=0.7):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

good_matches1_2 = filter_matches(matches1_2)
good_matches2_3 = filter_matches(matches2_3)
good_matches1_3 = filter_matches(matches1_3)

# 可视化特征点和匹配结果
def draw_matches(img1, kp1, img2, kp2, matches, title):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 10))
    plt.title(title)
    plt.imshow(img_matches, cmap='gray')
    plt.show()

# 可视化特征点匹配
draw_matches(img1, kp1, img2, kp2, good_matches1_2, "Image 1 & Image 2 Matches")
draw_matches(img2, kp2, img3, kp3, good_matches2_3, "Image 2 & Image 3 Matches")
draw_matches(img1, kp1, img3, kp3, good_matches1_3, "Image 1 & Image 3 Matches")
##############################################################################################################
# 函数：加载图像并进行检查
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img

# 提供图像路径
image_paths = [
    'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/1.jpg',
    'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/2.jpg',
    'D:\Pycharmprojects\lessons\Computer_Vision\photo_match/3.jpg'
]

# 加载图像
img1 = load_image(image_paths[0])
img2 = load_image(image_paths[1])
img3 = load_image(image_paths[2])

# 使用SIFT特征检测器
sift = cv2.SIFT_create()

# 检测特征点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

# 使用FLANN进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配特征点 (img1 <-> img2)
matches1_2 = flann.knnMatch(des1, des2, k=2)

# 匹配特征点 (img2 <-> img3)
matches2_3 = flann.knnMatch(des2, des3, k=2)

# 使用比值测试筛选匹配
def filter_matches(matches, ratio=0.7):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

good_matches1_2 = filter_matches(matches1_2)
good_matches2_3 = filter_matches(matches2_3)

# 提取好的匹配点
def extract_good_points(matches, kp1, kp2):
    points1 = []
    points2 = []
    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    return np.float32(points1), np.float32(points2)

# 提取匹配点
pts1_2, pts2_1 = extract_good_points(good_matches1_2, kp1, kp2)
pts2_3, pts3_2 = extract_good_points(good_matches2_3, kp2, kp3)

# 计算单应性矩阵
H1_2, mask1_2 = cv2.findHomography(pts1_2, pts2_1, cv2.RANSAC)
H2_3, mask2_3 = cv2.findHomography(pts2_3, pts3_2, cv2.RANSAC)

# 获取图像的尺寸
h1, w1 = img1.shape
h2, w2 = img2.shape
h3, w3 = img3.shape

# 获取图像的四个角点
corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
corners_img3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]]).reshape(-1, 1, 2)

# 变换角点
corners_img2_ = cv2.perspectiveTransform(corners_img2, H1_2)
corners_img3_ = cv2.perspectiveTransform(corners_img3, H1_2 @ H2_3)

# 拼接图像的边界尺寸
all_corners = np.concatenate((corners_img1, corners_img2_, corners_img3_), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

# 平移变换
translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

# 应用变换，进行图像拼接
result_img = cv2.warpPerspective(img1, H_translation @ np.eye(3), (x_max - x_min, y_max - y_min))
result_img[translation_dist[1]:h2 + translation_dist[1], translation_dist[0]:w2 + translation_dist[0]] = img2
result_img = cv2.warpPerspective(result_img, H_translation @ H1_2, (x_max - x_min, y_max - y_min))
result_img = cv2.warpPerspective(result_img, H_translation @ H1_2 @ H2_3, (x_max - x_min, y_max - y_min))
result_img[translation_dist[1]:h3 + translation_dist[1], translation_dist[0]:w3 + translation_dist[0]] = img3

# 显示拼接结果
plt.figure(figsize=(20, 10))
plt.title("Stitched Image")
plt.imshow(result_img, cmap='gray')
plt.axis('off')
plt.show()
#################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    """加载图像并转换为灰度图"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    return img

def detect_and_match_features(img1, img2):
    """检测并匹配特征点"""
    # 使用SIFT特征检测器和描述符
    sift = cv2.SIFT_create()

    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用FLANN进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 或者使用指定的检查数量

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 只保留那些通过比例测试的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches

def compute_fundamental_matrix(kp1, kp2, matches):
    """计算基础矩阵并剔除误匹配"""
    # 从匹配中提取对应点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 使用RANSAC算法估计基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # 只保留RANSAC内点
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    return F, inliers1, inliers2, mask

def draw_matches(img1, img2, kp1, kp2, matches, mask):
    """绘制匹配结果，突出显示内点"""
    # 只绘制内点匹配
    inlier_matches = [m for m, inl in zip(matches, mask) if inl]

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None, flags=2)
    return img_matches

# 主函数
if __name__ == '__main__':
    # 加载图像
    img1 = load_image('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/1.jpg')
    img2 = load_image('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/2.jpg')

    # 检测和匹配特征点
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    # 计算基础矩阵并处理误匹配
    F, inliers1, inliers2, mask = compute_fundamental_matrix(kp1, kp2, matches)

    print("Estimated Fundamental Matrix:")
    print(F)

    # 绘制匹配结果
    img_matches = draw_matches(img1, img2, kp1, kp2, matches, mask)

    # 显示匹配结果
    plt.figure(figsize=(15, 10))
    plt.title('Feature Matches with RANSAC Inliers')
    plt.imshow(img_matches, cmap='gray')
    plt.axis('off')
    plt.show()

###################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    """加载图像并转换为灰度图"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    return img

def detect_and_match_features(img1, img2):
    """检测并匹配特征点"""
    # 使用ORB特征检测器和描述符
    orb = cv2.ORB_create()

    # 检测关键点和计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 按照距离排序匹配
    matches = sorted(matches, key=lambda x: x.distance)

    return kp1, kp2, matches

def compute_fundamental_matrix(kp1, kp2, matches):
    """计算基础矩阵并剔除误匹配"""
    # 从匹配中提取对应点
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 使用RANSAC算法估计基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # 只保留RANSAC内点
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    return F, inliers1, inliers2, mask

def draw_matches(img1, img2, kp1, kp2, matches, mask):
    """绘制匹配结果，突出显示内点"""
    # 只绘制内点匹配
    inlier_matches = [m for m, inl in zip(matches, mask) if inl]

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None, flags=2)
    return img_matches

# 主函数
if __name__ == '__main__':
    # 加载图像
    img1 = load_image('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/1.jpg')
    img2 = load_image('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/2.jpg')

    # 检测和匹配特征点
    kp1, kp2, matches = detect_and_match_features(img1, img2)

    # 计算基础矩阵并处理误匹配
    F, inliers1, inliers2, mask = compute_fundamental_matrix(kp1, kp2, matches)

    print("Estimated Fundamental Matrix:")
    print(F)

    # 绘制匹配结果
    img_matches = draw_matches(img1, img2, kp1, kp2, matches, mask)

    # 显示匹配结果
    plt.figure(figsize=(15, 10))
    plt.title('Feature Matches with RANSAC Inliers')
    plt.imshow(img_matches)
    plt.show()
######################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_epilines(img1, img2, lines, pts1, pts2):
    """在图像上绘制极线"""
    r, c = img1.shape
    img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv2.circle(img1_color, tuple(map(int, pt1)), 5, color, -1)
        img2_color = cv2.circle(img2_color, tuple(map(int, pt2)), 5, color, -1)
    return img1_color, img2_color

# 主函数
if __name__ == '__main__':
    # 加载图像
    img1 = cv2.imread('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/1.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/2.jpg', cv2.IMREAD_GRAYSCALE)

    # 检测和匹配特征点
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 从匹配中提取点对
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # 计算基础矩阵
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # 计算并绘制极线
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines, _ = draw_epilines(img1, img2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines, _ = draw_epilines(img2, img1, lines2, pts2, pts1)

    # 显示极线图像
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img1_epilines)
    plt.subplot(122), plt.imshow(img2_epilines)
    plt.suptitle('Epipolar Lines')
    plt.show()

    # 极线校正
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w1, h1))

    # 校正图像
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    # 显示校正后的图像
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(img1_rectified, cmap='gray')
    plt.subplot(122), plt.imshow(img2_rectified, cmap='gray')
    plt.suptitle('Rectified Images')
    plt.show()
cv2.imwrite('ORB+BF1.jpg',img1_rectified)
cv2.imwrite('ORB+BF2.jpg',img2_rectified)
######################################################################################################################
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 函数：加载图像并进行检查
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img

# 提供图像路径
image_paths = [
    'D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/1.jpg',
    'D:\Pycharmprojects\lessons\Computer_Vision\photo_compute/2.jpg'
]

# 加载图像
img1 = load_image(image_paths[0])
img2 = load_image(image_paths[1])

# 使用SIFT特征检测器
sift = cv2.SIFT_create()

# 检测特征点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 使用FLANN进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配特征点 (img1 <-> img2)
matches = flann.knnMatch(des1, des2, k=2)

# 使用比值测试筛选匹配
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 提取匹配点对
points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 计算基础矩阵
F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

# 计算极线
def compute_epilines(points, which_image, F_matrix):
    if which_image == 1:
        epilines = cv2.computeCorrespondEpilines(points, 2, F_matrix)
    elif which_image == 2:
        epilines = cv2.computeCorrespondEpilines(points, 1, F_matrix)
    else:
        raise ValueError("which_image 参数必须是 1 或 2.")
    epilines = epilines.reshape(-1, 3)
    return epilines

# 计算图像1中点对应的极线
epilines1 = compute_epilines(points2, 2, F)

# 计算图像2中点对应的极线
epilines2 = compute_epilines(points1, 1, F)

# 极线矫正
h1, w1 = img1.shape
h2, w2 = img2.shape
_, H1, H2 = cv2.stereoRectifyUncalibrated(points1, points2, F, imgSize=(w1, h1))

# 应用矫正
rectified_img1 = cv2.warpPerspective(img1, H1, (w1, h1))
rectified_img2 = cv2.warpPerspective(img2, H2, (w2, h2))

# 显示矫正后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Rectified Image 1')
plt.imshow(rectified_img1, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Rectified Image 2')
plt.imshow(rectified_img2, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
# 将图像保存为新的文件
cv2.imwrite('SIFT+FLANN1.jpg',rectified_img1)
cv2.imwrite('SIFT+FLANN2.jpg',rectified_img2)
#############################################################################################################
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    return img

# 提供图像路径
image_paths = [
    'D:\\Pycharmprojects\\lessons\\Computer_Vision\\photo_match\\1.jpg',
    'D:\\Pycharmprojects\\lessons\\Computer_Vision\\photo_match\\2.jpg',
    'D:\\Pycharmprojects\\lessons\\Computer_Vision\\photo_match\\3.jpg'
]

# 加载图像
img1 = load_image(image_paths[0])
img2 = load_image(image_paths[1])
img3 = load_image(image_paths[2])

# 使用SIFT特征检测器
sift = cv2.SIFT_create()

# 检测特征点和描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
kp3, des3 = sift.detectAndCompute(img3, None)

# 使用FLANN进行特征点匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配特征点 (img1 <-> img2)
matches1_2 = flann.knnMatch(des1, des2, k=2)

# 匹配特征点 (img2 <-> img3)
matches2_3 = flann.knnMatch(des2, des3, k=2)

# 使用比值测试筛选匹配
def filter_matches(matches, ratio=0.75):
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

good_matches1_2 = filter_matches(matches1_2)
good_matches2_3 = filter_matches(matches2_3)

# 提取好的匹配点
def extract_good_points(matches, kp1, kp2):
    points1 = []
    points2 = []
    for match in matches:
        points1.append(kp1[match.queryIdx].pt)
        points2.append(kp2[match.trainIdx].pt)
    return np.float32(points1), np.float32(points2)

# 提取匹配点
pts1_2, pts2_1 = extract_good_points(good_matches1_2, kp1, kp2)
pts2_3, pts3_2 = extract_good_points(good_matches2_3, kp2, kp3)

# 计算单应性矩阵
H1_2, mask1_2 = cv2.findHomography(pts1_2, pts2_1, cv2.RANSAC)
H2_3, mask2_3 = cv2.findHomography(pts2_3, pts3_2, cv2.RANSAC)

# 获取图像的尺寸
h1, w1 = img1.shape
h2, w2 = img2.shape
h3, w3 = img3.shape

# 获取图像的四个角点
corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
corners_img3 = np.float32([[0, 0], [0, h3], [w3, h3], [w3, 0]]).reshape(-1, 1, 2)

# 变换角点
corners_img2_ = cv2.perspectiveTransform(corners_img2, H1_2)
corners_img3_ = cv2.perspectiveTransform(corners_img3, H1_2 @ H2_3)

# 拼接图像的边界尺寸
all_corners = np.concatenate((corners_img1, corners_img2_, corners_img3_), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

# 平移变换
translation_dist = [-x_min, -y_min]
H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

# 应用变换，进行图像拼接
result_img1 = cv2.warpPerspective(img1, H_translation @ np.eye(3), (x_max - x_min, y_max - y_min))
result_img2 = cv2.warpPerspective(img2, H_translation @ H1_2, (x_max - x_min, y_max - y_min))
result_img3 = cv2.warpPerspective(img3, H_translation @ H1_2 @ H2_3, (x_max - x_min, y_max - y_min))

# 转换为彩色图像，因为多频段融合需要彩色图像
result_img1_color = cv2.cvtColor(result_img1, cv2.COLOR_GRAY2BGR)
result_img2_color = cv2.cvtColor(result_img2, cv2.COLOR_GRAY2BGR)
result_img3_color = cv2.cvtColor(result_img3, cv2.COLOR_GRAY2BGR)

# 创建蒙版，定义重叠区域
mask1 = np.zeros_like(result_img1_color, dtype=np.uint8)
mask1[:h1, :w1] = 255

mask2 = np.zeros_like(result_img2_color, dtype=np.uint8)
mask2[:h2, :w2] = 255

mask3 = np.zeros_like(result_img3_color, dtype=np.uint8)
mask3[:h3, :w3] = 255

# 定义多频段融合函数
def multiband_blending(img1, img2, mask):
    # 创建图像金字塔
    g1 = img1.copy()
    g2 = img2.copy()
    gm = mask.copy()
    gp1 = [g1]
    gp2 = [g2]
    gpm = [gm]

    for i in range(6):
        g1 = cv2.pyrDown(g1)
        g2 = cv2.pyrDown(g2)
        gm = cv2.pyrDown(gm)
        gp1.append(g1)
        gp2.append(g2)
        gpm.append(gm)

    # 创建拉普拉斯金字塔
    lp1 = [gp1[5]]
    lp2 = [gp2[5]]
    gp_mask = [gpm[5]]

    for i in range(5, 0, -1):
        GE1 = cv2.pyrUp(gp1[i])
        # 调整尺寸以匹配目标图像
        GE1 = cv2.resize(GE1, (gp1[i-1].shape[1], gp1[i-1].shape[0]))
        L1 = cv2.subtract(gp1[i-1], GE1)
        lp1.append(L1)

        GE2 = cv2.pyrUp(gp2[i])
        # 调整尺寸以匹配目标图像
        GE2 = cv2.resize(GE2, (gp2[i-1].shape[1], gp2[i-1].shape[0]))
        L2 = cv2.subtract(gp2[i-1], GE2)
        lp2.append(L2)

        GE_mask = cv2.pyrUp(gpm[i])
        # 调整尺寸以匹配目标图像
        GE_mask = cv2.resize(GE_mask, (gpm[i-1].shape[1], gpm[i-1].shape[0]))
        GM = cv2.subtract(gpm[i-1], GE_mask)
        gp_mask.append(GM)

    # 合并金字塔
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gp_mask):
        ls = l1 * gm + l2 * (1 - gm)
        LS.append(ls)

    # 重建图像
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))  # 调整尺寸以匹配目标图像
        ls_ = cv2.add(ls_, LS[i])

    return ls_

# 使用多频段融合
blend1_2 = multiband_blending(result_img1_color, result_img2_color, mask2)
final_result = multiband_blending(blend1_2, result_img3_color, mask3)

# 转换为灰度图像以保持一致性
final_result_gray = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)

# 显示拼接结果
plt.figure(figsize=(20, 10))
plt.title("Stitched Image with Multiband Blending")
plt.imshow(final_result_gray, cmap='gray')
plt.axis('off')
plt.show()

############################################################################################################
# 函数：加载图像并进行检查
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像未找到: {path}")
    if img.dtype != np.uint8:
        img = cv2.convertScaleAbs(img)
    return img

# 提供平行视图路径
rectified_paths = [
    'D:\Pycharmprojects\lessons\Computer_Vision\Rectified_images2\ORB+BF1.jpg',
    'D:\Pycharmprojects\lessons\Computer_Vision\Rectified_images2\ORB+BF2.jpg'
]

# 加载平行视图
rectified_img1 = load_image(rectified_paths[0])
rectified_img2 = load_image(rectified_paths[1])

# 使用SGBM进行视差计算
min_disp = 0
num_disp = 128 - min_disp
window_size = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size ** 2,
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=200,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

# 计算视差图
disparity = stereo.compute(rectified_img1, rectified_img2)

# 视差图转换为深度图
depth_map = np.zeros(disparity.shape, dtype=np.float32)
Q = np.float32([[1, 0, 0, -rectified_img1.shape[1] / 2],
                [0, 1, 0, -rectified_img1.shape[0] / 2],
                [0, 0, 0, rectified_img1.shape[1] * 0.7],
                [0, 0, -1. / 3.5, 0]])

depth_map = cv2.reprojectImageTo3D(disparity.astype(np.float32)/16., Q)

# 显示视差图和深度图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Disparity Map')
plt.imshow(disparity, cmap='plasma')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title('Depth Map')
plt.imshow(depth_map[:, :, 2], cmap='plasma')
plt.colorbar()

plt.tight_layout()
plt.show()
######################################################################################################
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np

class ImageStitchingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Stitching App with Poisson Blending')
        self.setGeometry(100, 100, 800, 600)

        # 创建选择图像的按钮
        self.select_button = QPushButton('选择图像', self)
        self.select_button.setGeometry(20, 20, 120, 30)
        self.select_button.clicked.connect(self.select_image)

        # 创建拼接按钮
        self.stitch_button = QPushButton('拼接图像', self)
        self.stitch_button.setGeometry(160, 20, 120, 30)
        self.stitch_button.clicked.connect(self.stitch_images)

        # 创建用于显示图像的标签
        self.image_label = QLabel(self)
        self.image_label.setGeometry(20, 70, 760, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")  # 添加边框以便观察图像大小

        # 创建进度条
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(20, 490, 760, 20)
        self.progress_bar.setValue(0)

        # 布局管理
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.stitch_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.progress_bar)

        # 初始化图像列表
        self.images = []

    def select_image(self):
        if len(self.images) >= 3:
            print("已经选择了三张图像。")
            return

        # 弹出文件选择对话框
        file, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "图像文件 (*.jpg *.png *.bmp *.jpeg)")

        if not file:
            print("未选择图像文件。")
            return

        image = cv2.imread(file)
        if image is None:
            print(f"无法读取图像：{file}")
            return

        # 调整图像大小以适合拼接
        resized_image = cv2.resize(image, (400, 300))  # 调整为相同的大小，这个大小可以根据需要调整
        self.images.append(resized_image)
        print(f"添加图像：{file}")

        # 更新进度条
        self.update_progress(len(self.images) * 33)  # 假设每添加一张图像进度增加33%

    def stitch_images(self):
        if len(self.images) != 3:
            print("请选择三张图像进行拼接！")
            return

        # 图像拼接处理
        self.progress_bar.setValue(50)  # 拼接过程开始，设置进度为50%
        stitched_image = self.opencv_stitch(self.images)

        if stitched_image is not None:
            self.progress_bar.setValue(75)  # 拼接完成，设置进度为75%

            # 图像融合处理（泊松融合）
            blended_image = self.poisson_blend(self.images, stitched_image)
            self.progress_bar.setValue(100)  # 融合完成，设置进度为100%

            # 显示融合后的图像
            self.display_image(blended_image)

    def opencv_stitch(self, images):
        # 使用OpenCV的Stitcher类进行拼接
        stitcher = cv2.Stitcher_create()
        status, stitched = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            print("拼接成功")
            return stitched
        else:
            print(f"拼接失败，状态码：{status}")
            return None

    def poisson_blend(self, images, stitched_img):
        if len(images) != 3:
            print("需要三张图像进行泊松融合。")
            return stitched_img

        img1, img2, img3 = images

        # 将OpenCV的图像格式转换为适合泊松融合的格式
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)

        # 创建掩膜，用于指示泊松融合的区域
        mask1 = np.zeros_like(img1[:, :, 0], dtype=np.uint8)
        mask2 = np.zeros_like(img2[:, :, 0], dtype=np.uint8)
        mask1[:, img1.shape[1] // 2:] = 255  # 从中间分割的掩膜
        mask2[:, img2.shape[1] // 2:] = 255  # 从中间分割的掩膜

        # 融合img1和img2
        blended_1_2 = cv2.seamlessClone(img1, img2, mask1, (img1.shape[1] // 2, img1.shape[0] // 2), cv2.NORMAL_CLONE)

        # 融合img2和img3
        blended_2_3 = cv2.seamlessClone(blended_1_2, img3, mask2, (blended_1_2.shape[1] // 2, blended_1_2.shape[0] // 2), cv2.NORMAL_CLONE)

        return blended_2_3

    def display_image(self, image):
        if image is not None:
            # 将OpenCV图像转换为Qt可显示的格式
            height, width, channel = image.shape
            bytesPerLine = 3 * width
            qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)
        else:
            print("无法显示图像。")

    def update_progress(self, value):
        self.progress_bar.setValue(value)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageStitchingApp()
    ex.show()
    sys.exit(app.exec_())
