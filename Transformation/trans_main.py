import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    img = cv2.imread('flash.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    print(h, w)

    # Translation
    T = np.array([
        [1, 0, 100],
        [0, 1, -30],
        [0, 0, 1],
    ])

    # Scale
    S = np.array([
        [1.5, 0, 0],
        [0, .7, 0],
        [0, 0, 1],
    ])

    # Rotation
    R = np.eye(3)
    R[:2, :] = cv2.getRotationMatrix2D((0, 0), 45, 1)

    # Mix and match your transformation
    m = T.dot(R).dot(S)

    t = np.eye(3)
    t[:2, -1] = [w / 2, h / 2]

    # Centers the image for scale/rotation transformation
    m = t.dot(m).dot(np.linalg.inv(t))
    m = np.linalg.inv(m)

    xs, ys = np.meshgrid(range(w), range(h))
    zs = np.ones_like(xs)
    xyz = np.array([xs.flatten(), ys.flatten(), zs.flatten()])

    xyz_t = m.dot(xyz)[:-1, :].astype(int)

    # Edge reflect
    # xyz_t[0, xyz_t[0, :] >= w - 1] = w - 1
    # xyz_t[1, xyz_t[1, :] >= h - 1] = h - 1
    # xyz_t[xyz_t < 0] = 0

    # Edge wrap
    xyz_t[0, :] = np.mod(xyz_t[0, :], w - 1)
    xyz_t[1, :] = np.mod(xyz_t[1, :], h - 1)

    img_t = img[xyz_t[1, :], xyz_t[0, :]].reshape((h, w, -1))

    f, ax = plt.subplots(1, 2)
    ax[0].imshow(img)
    ax[1].imshow(img_t)

    plt.show()


if __name__ == '__main__':
    main()
