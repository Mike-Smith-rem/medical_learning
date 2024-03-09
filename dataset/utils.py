'''Copyright oyk
Created 28 17:41:45
'''
if __name__ == "__main__":
    import os, cv2
    path0 = "classifer/train100"
    path1 = "classifer/test100"

    img0 = "classifer/benign/img"
    img1 = "classifer/malignant/img"
    img2 = "classifer/normal/img"

    mask0 = "classifer/benign/mask"
    mask1 = "classifer/malignant/mask"
    mask2 = "classifer/normal/mask"

    img0s = os.listdir(img0)
    img1s = os.listdir(img1)
    img2s = os.listdir(img2)

    mask0s = os.listdir(mask0)
    mask1s = os.listdir(mask1)
    mask2s = os.listdir(mask2)

    train_size = 100
    # for i in range(train_size):
    #     _p0 = "classifer/train100/benign"
    #     _p1 = "classifer/train100/malignant"
    #     _p2 = "classifer/train100/normal"
    #
    #     if not os.path.exists(_p0) and not os.path.exists(_p1) and not os.path.exists(_p2):
    #         os.makedirs(_p0)
    #         os.makedirs(_p1)
    #         os.makedirs(_p2)
    #
    #     p0 = os.path.join(img0, img0s[i])
    #     p1 = os.path.join(img1, img1s[i])
    #     p2 = os.path.join(img2, img2s[i])
    #
    #     i0 = cv2.imread(p0)
    #     i1 = cv2.imread(p1)
    #     i2 = cv2.imread(p2)
    #
    #     w0 = os.path.join(_p0, img0s[i])
    #     w1 = os.path.join(_p1, img1s[i])
    #     w2 = os.path.join(_p2, img2s[i])
    #
    #     cv2.imwrite(w0, i0)
    #     cv2.imwrite(w1, i1)
    #     cv2.imwrite(w2, i2)

    test0_size = len(img0s) - train_size
    test1_size = len(img1s) - train_size
    test2_size = len(img2s) - train_size

    test_size = min(test0_size, min(test1_size, test2_size))
    #
    # for i in range(train_size, train_size + test_size):
    #     _p0 = "classifer/test100/benign"
    #     _p1 = "classifer/test100/malignant"
    #     _p2 = "classifer/test100/normal"
    #     _p3 = "classifer/test100/all"
    #
    #     if not os.path.exists(_p0) and not os.path.exists(_p1) and not os.path.exists(_p2):
    #         os.makedirs(_p0)
    #         os.makedirs(_p1)
    #         os.makedirs(_p2)
    #
    #     p0 = os.path.join(img0, img0s[i])
    #     p1 = os.path.join(img1, img1s[i])
    #     p2 = os.path.join(img2, img2s[i])
    #
    #     i0 = cv2.imread(p0)
    #     i1 = cv2.imread(p1)
    #     i2 = cv2.imread(p2)
    #
    #     w0 = os.path.join(_p0, img0s[i])
    #     w1 = os.path.join(_p1, img1s[i])
    #     w2 = os.path.join(_p2, img2s[i])
    #
    #     cv2.imwrite(w0, i0)
    #     cv2.imwrite(w1, i1)
    #     cv2.imwrite(w2, i2)

    seg_train_size = 100
    p_img = "classifer/train100/all/img"
    p_mask = "classifer/train100/all/mask"

    d_img = "classifer/test100/all/img"
    d_mask = "classifer/test100/all/mask"

    if not os.path.exists(p_img) and not os.path.exists(p_mask) \
        and not os.path.exists(d_img) and not os.path.exists(d_mask):
        os.makedirs(p_img)
        os.makedirs(p_mask)
        os.makedirs(d_img)
        os.makedirs(d_mask)

    for i in range(seg_train_size):
        i0_name = img0s[i]
        m0_name = mask0s[i]

        _p0_img = "classifer/benign/img"
        _p0_mask = "classifer/benign/mask"
        img_0 = cv2.imread(os.path.join(_p0_img, i0_name))
        mask_0 = cv2.imread(os.path.join(_p0_mask, m0_name))

        cv2.imwrite(os.path.join(p_img, '0_' + i0_name), img_0)
        cv2.imwrite(os.path.join(p_mask, '0_' + m0_name), mask_0)

        i1_name = img1s[i]
        m1_name = mask1s[i]

        _p1_img = "classifer/malignant/img"
        _p1_mask = "classifer/malignant/mask"

        img_1 = cv2.imread(os.path.join(_p1_img, i1_name))
        mask_1 = cv2.imread(os.path.join(_p1_mask, m1_name))

        cv2.imwrite(os.path.join(p_img, '1_' + i1_name), img_1)
        cv2.imwrite(os.path.join(p_mask, '1_' + m1_name), mask_1)

        i2_name = img2s[i]
        m2_name = mask2s[i]

        _p2_img = "classifer/normal/img"
        _p2_mask = "classifer/normal/mask"

        img_2 = cv2.imread(os.path.join(_p2_img, i2_name))
        mask_2 = cv2.imread(os.path.join(_p2_mask, m2_name))

        cv2.imwrite(os.path.join(p_img, '2_' + i2_name), img_2)
        cv2.imwrite(os.path.join(p_mask, '2_' + m2_name), mask_2)

    seg_test_size = test_size
    for i in range(seg_train_size, seg_train_size + seg_test_size):
        i0_name = img0s[i]
        m0_name = mask0s[i]

        _p0_img = "classifer/benign/img"
        _p0_mask = "classifer/benign/mask"
        img_0 = cv2.imread(os.path.join(_p0_img, i0_name))
        mask_0 = cv2.imread(os.path.join(_p0_mask, m0_name))

        cv2.imwrite(os.path.join(d_img, '0_' + i0_name), img_0)
        cv2.imwrite(os.path.join(d_mask, '0_' + m0_name), mask_0)

        i1_name = img1s[i]
        m1_name = mask1s[i]

        _p1_img = "classifer/malignant/img"
        _p1_mask = "classifer/malignant/mask"

        img_1 = cv2.imread(os.path.join(_p1_img, i1_name))
        mask_1 = cv2.imread(os.path.join(_p1_mask, m1_name))

        cv2.imwrite(os.path.join(d_img, '1_' + i1_name), img_1)
        cv2.imwrite(os.path.join(d_mask, '1_' + m1_name), mask_1)

        i2_name = img2s[i]
        m2_name = mask2s[i]

        _p2_img = "classifer/normal/img"
        _p2_mask = "classifer/normal/mask"

        img_2 = cv2.imread(os.path.join(_p2_img, i2_name))
        mask_2 = cv2.imread(os.path.join(_p2_mask, m2_name))

        cv2.imwrite(os.path.join(d_img, '2_' + i2_name), img_2)
        cv2.imwrite(os.path.join(d_mask, '2_' + m2_name), mask_2)

    print("Finished!")
