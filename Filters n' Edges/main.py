import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import match_template


def filterBasic() -> None:
    """
        Toy example on filtering
        Blurring an image using a Box-Filter
        :return:
    """
    img_path = 'scarlet_witch.png'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    k_size = 11
    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum()

    img_no_vision = cv2.filter2D(img, -1, kernel)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(img_no_vision)

    plt.show()


def filterMedian() -> None:
    """
        Displaying median filter vs mean filter, and it's affect on edges
        :return:
    """
    img_path = 'median.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    k_size = 5
    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum()

    box_filt = cv2.filter2D(img, -1, kernel)
    median_filt = cv2.medianBlur(img, k_size)

    f, ax = plt.subplots(2, 3)
    [axi.set_axis_off() for axi in ax.ravel()]
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(box_filt)
    ax[0, 2].imshow(median_filt)

    dev_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    ax[1, 0].imshow(cv2.filter2D(img, -1, dev_kernel))
    ax[1, 1].imshow(cv2.filter2D(box_filt, -1, dev_kernel))
    ax[1, 2].imshow(cv2.filter2D(median_filt, -1, dev_kernel))
    plt.show()


def corrSimple(use_nnc=False) -> None:
    """
        Basic correlation example and Normalized cross correlation
        :return:
    """
    img = np.zeros((800, 800))
    img = cv2.circle(img, (250, 300), 30, (1, 1, 1), -1)
    img = cv2.circle(img, (500, 400), 60, (1, 1, 1), -1)
    img = cv2.rectangle(img, (500, 200), (540, 300), (1, 1, 1), -1)

    plt.imshow(img)
    plt.show()

    corr_kernel = np.ones((150, 150)) * 0
    corr_kernel = cv2.circle(corr_kernel, (75, 75), 30, (1, 1, 1), -1)

    if use_nnc:
        nose_img = match_template(img, corr_kernel, pad_input=True, mode='symmetric')
    else:
        nose_img = cv2.filter2D(img, -1, corr_kernel, borderType=cv2.BORDER_REFLECT)

    max_resp = nose_img.max()
    y, x = np.where(nose_img == max_resp)

    f, ax = plt.subplots(2, 3)
    plt.set_cmap('hot')
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(corr_kernel, vmin=0, vmax=1)
    ax[0, 2].imshow(nose_img)

    ax[1, 0].imshow(img)
    ax[1, 0].plot(x, y, 'gX')

    ax[1, 2].imshow(nose_img)
    ax[1, 2].plot(x, y, 'gX')
    plt.show()


def corrAdv(use_ncc=False) -> None:
    """
        Basic correlation example and Normalized cross correlation
        :return:
    """
    face_path = 'face.png'
    img_face = cv2.imread(face_path, cv2.IMREAD_GRAYSCALE)
    img_face = cv2.resize(img_face, (0, 0), fx=.25, fy=.25) / 255

    nose_kernel = np.ones((50, 120)) * (0)
    k_h, k_w = nose_kernel.shape
    n_width = 30
    nose_kernel[:, k_w // 2 - n_width:k_w // 2 + n_width] = 1
    nose_kernel = cv2.imread('eye.jpg', cv2.IMREAD_GRAYSCALE)
    nose_kernel = cv2.resize(nose_kernel, (0, 0), fx=0.04, fy=0.04)

    if use_ncc:
        nose_img = match_template(img_face, nose_kernel, pad_input=True, mode='symmetric')
    else:
        nose_img = cv2.filter2D(img_face, -1, nose_kernel, borderType=cv2.BORDER_REFLECT)
    max_resp = nose_img.max()
    y, x = np.where(nose_img == max_resp)

    f, ax = plt.subplots(2, 3)
    plt.set_cmap('hot')
    ax[0, 0].imshow(img_face)
    ax[0, 1].imshow(nose_kernel)
    ax[0, 2].imshow(nose_img)

    ax[1, 0].imshow(img_face)
    ax[1, 0].plot(x, y, 'gX')

    ax[1, 2].imshow(nose_img)
    ax[1, 2].plot(x, y, 'gX')
    plt.show()


def step3D(x, n):
    return 1 * np.arctan(x) + n * np.random.standard_normal(x.shape)


def disp(X: np.ndarray, Y: np.ndarray, Z1: np.ndarray, Z2: np.ndarray = None):
    if Z2 is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    else:
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_title('Sobel')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(fun_range, -fun_range)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.plot_surface(X, Y, Z2, rstride=1, cstride=1,
                        cmap='viridis', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    ax.set_xlim(fun_range, -fun_range)
    plt.show()


def filterSobel() -> None:
    """
        Sobel filter example
        :return:
    """
    global fun_range
    fun_range = 10
    dpi = 50
    x = np.linspace(-fun_range, fun_range, dpi)
    y = np.linspace(-fun_range, fun_range, dpi)

    X, Y = np.meshgrid(x, y)

    # Clean
    Z = step3D(X, 0)
    # disp(X, Y, Z, None)

    # With noise
    np.random.seed(42)
    Z_noise = step3D(X, 1)
    # disp(X, Y, Z_noise, None)

    k_size = 3
    bk_size = 21
    # Smooth
    kernel = np.ones((bk_size, k_size))
    kernel /= kernel.sum()
    Z_smooth = cv2.filter2D(Z_noise, -1, kernel, borderType=cv2.BORDER_REFLECT)

    # Sobol
    kernel_abs_sob = np.ones((bk_size, k_size))
    kernel_abs_sob[:, 1] = 0
    kernel_abs_sob = kernel_abs_sob / kernel_abs_sob.sum()
    Z_sobol = cv2.filter2D(Z_noise, -1, kernel_abs_sob, borderType=cv2.BORDER_REFLECT)

    f, ax = plt.subplots(1, 3)
    ax[0].imshow(Z_sobol, vmin=-3, vmax=3)
    ax[1].imshow(Z_noise, vmin=-3, vmax=3)
    ax[2].imshow(Z_smooth, vmin=-3, vmax=3)
    plt.show()
    # disp(X, Y, Z_sobol, Z_smooth)

    kernel_der = np.zeros((k_size, 3))
    kernel_der[:, 0] = 1
    kernel_der[:, -1] = -1
    kernel_sob = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ])

    # Smooth
    kernel = np.ones((k_size, k_size))
    kernel /= kernel.sum()
    Z_smooth_rg = cv2.filter2D(Z, -1, kernel_der, borderType=cv2.BORDER_REFLECT)
    Z_smooth_derv = cv2.filter2D(Z_smooth, -1, kernel_der, borderType=cv2.BORDER_REFLECT)
    Z_sobol_derv = cv2.filter2D(Z_sobol, -1, kernel_der, borderType=cv2.BORDER_REFLECT)
    # disp(X, Y, Z_sobol_derv, Z_smooth_derv)
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(Z_sobol_derv, vmin=-3, vmax=3)
    ax[1].imshow(Z_smooth_rg, vmin=-3, vmax=3)
    ax[2].imshow(Z_smooth_derv, vmin=-3, vmax=3)
    plt.show()


def edgeDetect() -> None:
    """
        Edge detection example
        :return:
    """
    img = np.zeros((800, 800))
    img = cv2.circle(img, (250, 300), 30, (1, 1, 1), -1)
    img = cv2.circle(img, (500, 400), 60, (1, 1, 1), -1)
    img = cv2.rectangle(img, (500, 200), (540, 300), (1, 1, 1), -1)

    plt.imshow(img)
    plt.show()

    kernel = np.array([[1, 0, -1]])
    dx = cv2.filter2D(img, -1, kernel[:, ::-1])
    dy = cv2.filter2D(img, -1, kernel[:, ::-1].T)
    grad = np.sqrt(dx ** 2 + dy ** 2)
    dir_mat = np.degrees(np.arctan2(dy, dx))

    f, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(dx)
    ax[0, 1].imshow(dy)
    ax[1, 0].imshow(grad)
    ax[1, 1].imshow(dir_mat)
    plt.show()


def main():
    # filterBasic()
    # filterMedian()
    # corrSimple()
    # corrAdv(use_ncc=False)
    # corrAdv(use_ncc=True)
    edgeDetect()
    # filterSobel()


if __name__ == '__main__':
    main()
