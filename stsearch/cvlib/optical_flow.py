import typing

import cv2
from logzero import logger
import numpy as np
from sklearn.cluster import KMeans

# adapted idea fomr https://github.com/jguoaj/multi-object-tracking
# but re-implemented. Result is worse than theirs.
# Reason seems to be optical flow.

# params for ShiTomasi corner detection
FEATURE_PARAMS = dict(
    maxCorners=50,
    qualityLevel=0.3,
    minDistance=3,
    blockSize=7)

LK_PARAMS = dict(
    winSize  = (15, 15),
    maxLevel = 3,
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def get_good_features_to_track(
        img_gray: np.ndarray, bboxs: np.ndarray) -> typing.List[np.ndarray]:
    """[summary]

    Args:
        img_gray (np.ndarray): A gray scale image
        bboxs (np.ndarray): n_box x 4. Each row (xmin, ymin, xmax, ymax)

    Returns:
        typing.List[nd.ndarray]: A list of len n_box. Each element is n_feature x 2 (x, y).
    """

    box_features = []
    for xmin, ymin, xmax, ymax in bboxs:
        p = cv2.goodFeaturesToTrack(
            img_gray[ymin:ymax, xmin:xmax], mask=None, **FEATURE_PARAMS)
        if p is not None:
            p = p.reshape((-1, 2))
            p[:, 0] += xmin
            p[:, 1] += ymin
            box_features.append(p)
        else:
            box_features.append(np.empty((0, 2)))

    assert len(box_features) == bboxs.shape[0]
    return box_features


def estimate_feature_translation(
    img_old: np.ndarray,
    img_new: np.ndarray,
    box_features: typing.List[np.ndarray]
    ) -> typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]:
    """[summary]

    Args:
        img_old (np.ndarray): [description]
        img_new (np.ndarray): [description]
        box_features (typing.List[np.ndarray]): [description]

    Returns:
        typing.Tuple[typing.List[np.ndarray], typing.List[np.ndarray]]: 
        Good old features and good new features, respectively. Bad features will
        be removed.
    """
    n_old_features = [f.shape[0] for f in box_features]
    p0 = np.vstack(box_features).astype(np.float32)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img_old, img_new, p0.reshape((-1, 1, 2)), None, **LK_PARAMS)

    if p1 is None:
        return [np.empty((0,2)) for _ in box_features ],  [np.empty((0,2)) for _ in box_features ]
    p1 = p1.reshape((-1, 2))
    st = st.reshape((-1,))
    assert p0.shape == p1.shape
    good_old_features = []
    good_new_features = []
    for n in n_old_features:
        good_old_features.append(p0[:n][st[:n]==1])
        good_new_features.append(p1[:n][st[:n]==1])
        # remove the first n elements
        p0 = p0[n:]
        p1 = p1[n:]
        st = st[n:]

    assert len(good_old_features) == len(good_new_features)
    return good_old_features, good_new_features


def estimate_box_translation(
    old_features: typing.List[np.ndarray],
    new_features: typing.List[np.ndarray],
    old_bboxs: np.ndarray
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
    """[summary]

    Args:
        old_features (typing.List[np.ndarray]): [description]
        new_features (typing.List[np.ndarray]): [description]
        old_bboxs (np.ndarray): n_box x 4 (xmin, ymin, xmax, ymax)

    Returns:
        typing.Tuple[np.ndarray, np.ndarray]: new_bboxs, status
    """
    status = []
    new_bboxs = []
    for old_pts, new_pts, (xmin, ymin, xmax, ymax) in zip(old_features, new_features, old_bboxs):
        assert old_pts.shape == new_pts.shape

        # filter pts that are in the box
        MARGIN=2
        in_box_inds = (old_pts[:,0]>=xmin-MARGIN) & (old_pts[:,1]>=ymin-MARGIN) & (old_pts[:,0]<=xmax+MARGIN) & (old_pts[:,1]<=ymax+MARGIN)
        old_pts = old_pts[in_box_inds]
        new_pts = new_pts[in_box_inds]

        if old_pts.shape[0] <= 1:
            logger.warn("Too few points for estimating box")
            new_bboxs.append([-1,-1,-1,-1])
            status.append(0)
            continue

        pts_shift = new_pts - old_pts
        median_shift = np.median(pts_shift, axis=0)

        # choose majority that project positive onto median_shift
        majority_inds = (pts_shift @ median_shift) > 0.0

        # choose majority that's withink threshold difference  to median shift
        # THRES = 3
        # majority_inds = np.linalg.norm(pts_shift - median_shift, axis=1) <= THRES

        # choose 1/2 pts closest to median
        # diff_from_median = np.linalg.norm(pts_shift - median_shift, axis=1)
        # majority_inds= diff_from_median <= np.median(diff_from_median)

        if np.count_nonzero(majority_inds) > 2:
            majority_old_pts = old_pts[majority_inds]
            majority_new_pts = new_pts[majority_inds]
        else:
            majority_old_pts = old_pts
            majority_new_pts = new_pts

        # # if all points shift in the same x/y directions, we don't do kmeans
        # if (np.multiply(pts_shift.min(axis=0), pts_shift.max(axis=0)) >= 0).all():
        #     majority_old_pts = old_pts
        #     majority_new_pts = new_pts
        # else:
        #     kmeans = KMeans(n_clusters=2, random_state=0).fit(pts_shift)
        #     if np.count_nonzero(kmeans.labels_==0) >= np.count_nonzero(kmeans.labels_==1):
        #         majority = 0
        #     else:
        #         majority = 1        
        #     majority_old_pts = old_pts[kmeans.labels_ == majority]
        #     majority_new_pts = new_pts[kmeans.labels_ == majority]

        M, mask = cv2.estimateAffinePartial2D(
        # M, mask = cv2.estimateAffine2D(
            majority_old_pts.reshape((-1,1,2)),
            majority_new_pts.reshape((-1,1,2)), 
            method=cv2.RANSAC)

        if np.count_nonzero(mask) <= 2:
            # logger.warn("Too few inliner in Affine estiamte")
            new_bboxs.append([-1,-1,-1,-1])
            status.append(0)
        else:

            # re-estimate a second time with stricter inliners
            P_THRESH = 1.5
            projection = cv2.transform(majority_old_pts.reshape((-1,1,2)), M).reshape((-1,2))
            perror = np.linalg.norm(majority_new_pts - projection, axis=1)  # n_pts
            inliner_inds = perror < P_THRESH
            if np.count_nonzero(inliner_inds) >= 4:
                try:
                    M, mask = cv2.estimateAffinePartial2D(
                        majority_old_pts[inliner_inds].reshape((-1,1,2)),
                        majority_new_pts[inliner_inds].reshape((-1,1,2)),
                        method=cv2.RANSAC
                    )
                except:
                    print(majority_inds)
                    print(majority_old_pts)
                    print(majority_new_pts)
                    raise

            # translate bbox
            src = np.array([[xmin, ymin], [xmax, ymax]])
            dst = cv2.transform(src.reshape((-1,1,2)), M)
            (new_xmin, new_ymin), (new_xmax, new_ymax) = dst.reshape((-1,2))

            # # box size shouldn't change abruptly
            # SIZE_CHANGE_THRESH = 1.5
            # if not (1/SIZE_CHANGE_THRESH < abs(new_xmax-new_xmin) /  abs(xmax-xmin) < SIZE_CHANGE_THRESH and \
            #     1/SIZE_CHANGE_THRESH < abs(new_ymax-new_ymin) / abs(ymax-ymin) < SIZE_CHANGE_THRESH):
            #     # simply do mean shift can keep box size
            #     new_xmin, new_ymin = median_shift + [xmin, ymin]
            #     new_xmax, new_ymax = median_shift + [xmax, ymax]

            new_bboxs.append([new_xmin, new_ymin, new_xmax, new_ymax])
            status.append(1)

    return np.array(new_bboxs), np.array(status)        



