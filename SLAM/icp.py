"""
Iterative Closest Point(ICP) SLAM
"""
import math
import matplotlib.pyplot as plt
import numpy as np

# Icp Parameters
EPS = 0.0001
MAX_ITER = 100

show_animation = True

def icp_matching(previous_points, current_points):
    """
    :param previous_points: 2d points in the previous frame
    :param current_points:  2d points in the current frame
    :return: R:Rotation matrix T:Translation vector
    """
    H = None    # homogeneous transformation matrix

    dError = 1000.0
    preError = 1000.0
    count = 0

    while dError >= EPS:
        count += 1

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect("key_release_event",
                                         lambda event : [exit(0) if event.key == "escape" else None])
            plt.plot(previous_points[0, :], previous_points[1, :], ".r")
            plt.plot(current_points[0, :], current_points[1, :], ".b")
            plt.plot(0.0, 0.0, "xr")
            plt.axis("equal")
            plt.pause(0.1)

        # 计算变换前后各个点对应的最近点索引和整体误差
        indexes, error = nearest_neighbor_association(previous_points, current_points)
        # 使用svd计算变换和平移矩阵
        Rt, Tt = svd_motion_estimation(previous_points[:, indexes], current_points)

        # 更新变换后的点
        current_points = (Rt @ current_points) + Tt[:, np.newaxis]

        # 更新H矩阵
        H = update_homogeneous_matrix(H, Rt, Tt)

        # 更新计算误差
        dError = abs(preError - error)
        preError = error
        print("Residual:", error)

        if dError <= EPS:
            print("Converge", error, dError, count)
            break
        elif MAX_ITER <= count:
            print("Not Converge...", error, dError, count)
            break

    # 提取变换和平移矩阵
    R = np.array(H[0 : 2, 0 : 2])
    T = np.array(H[0 : 2, 2])

    return R, T


def update_homogeneous_matrix(Hin, R, T):
    H = np.zeros((3, 3))

    H[0, 0] = R[0, 0]
    H[1, 0] = R[1, 0]
    H[0, 1] = R[0, 1]
    H[1, 1] = R[1, 1]
    H[2, 2] = 1.0

    H[0, 2] = T[0]
    H[1, 2] = T[1]

    if Hin is None:
        return H
    else:
        return Hin @ H


def nearest_neighbor_association(previous_points, current_points):
    # calc the sum of residual errors
    delta_points = previous_points - current_points
    d = np.linalg.norm(delta_points, axis=0)
    error = sum(d)

    # calc index with nearest neighbor assosiation
    d = np.linalg.norm(
        np.repeat(current_points, previous_points.shape[1], axis=1)
        - np.tile(previous_points, (1, current_points.shape[1])), axis=0)
    indexes = np.argmin(
        d.reshape(current_points.shape[1], previous_points.shape[1]), axis=1)

    return indexes, error


def svd_motion_estimation(previous_points, current_points):
    # 按行取平均值
    pm = np.mean(previous_points, axis=1)
    cm = np.mean(current_points, axis=1)
    # 取中心化处理
    p_shift = previous_points - pm[:, np.newaxis]
    c_shift = current_points - cm[:, np.newaxis]

    # 计算W矩阵
    W = c_shift @ p_shift.T
    # 对W进行svd分解
    u, s, vh = np.linalg.svd(W)

    # 计算R和T
    R = (u @ vh).T
    t = pm - (R @ cm)

    return R, t


def main():
    # 仿真参数
    nPoint = 1000                           # 点个数
    fieldLength = 50.0                      # 缩放尺度
    motion = [0.5, 2.0, np.deg2rad(-10.0)]  # 平移和变换角 [x[m],y[m],yaw[deg]]

    nsim = 3                                # 仿真次数

    for _ in range(nsim):
        # 变换前的点
        px = (np.random.rand(nPoint) - 0.5) * fieldLength
        py = (np.random.rand(nPoint) - 0.5) * fieldLength
        previous_points = np.vstack((px, py))

        # 变换后的点
        cx = [math.cos(motion[2]) * x - math.sin(motion[2]) * y + motion[0]
              for (x, y) in zip(px, py)]
        cy = [math.sin(motion[2]) * x + math.cos(motion[2]) * y + motion[1]
              for (x, y) in zip(px, py)]
        current_points = np.vstack((cx, cy))

        # 计算变换和平移
        R, T = icp_matching(previous_points, current_points)
        print("R:", R)
        print("T:", T)


if __name__ == "__main__":
    main()
