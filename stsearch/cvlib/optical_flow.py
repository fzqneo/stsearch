import typing

import cv2
from logzero import logger
import numpy as np
from sklearn.cluster import KMeans

# params for ShiTomasi corner detection
FEATURE_PARAMS = dict(
    maxCorners=100,
    qualityLevel=0.3,
    minDistance=3,
    blockSize=7)

LK_PARAMS = dict(
    winSize  = (21, 21),
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
    p0 = np.vstack(box_features)
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        img_old, img_new, p0.reshape((-1, 1, 2)), None, **LK_PARAMS)

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
        pts_shift = new_pts - old_pts
        kmeans = KMeans(n_clusters=2, random_state=0).fit(pts_shift)
        if np.count_nonzero(kmeans.labels_==0) >= np.count_nonzero(kmeans.labels_==1):
            majority = 0
        else:
            majority = 1
        
        majority_old_pts = old_pts[kmeans.labels_ == majority]
        majority_new_pts = new_pts[kmeans.labels_ == majority]

        M, mask = cv2.estimateAffinePartial2D(
            majority_old_pts.reshape((-1,1,2)),
            majority_new_pts.reshape((-1,1,2)), 
            method=cv2.RANSAC)

        if np.count_nonzero(mask) <= 2:
            logger.warn("Too few inliner in Affine estiamte")
            new_bboxs.append([-1,-1,-1,-1])
            status.append(0)
        else:
            # translate bbox
            src = np.array([[xmin, ymin], [xmax, ymax]])
            dst = cv2.transform(src.reshape((-1,1,2)), M)
            (new_xmin, new_ymin), (new_xmax, new_ymax) = dst.reshape((-1,2))
            new_bboxs.append([new_xmin, new_ymin, new_xmax, new_ymax])
            status.append(1)

    return np.array(new_bboxs), np.array(status)        



