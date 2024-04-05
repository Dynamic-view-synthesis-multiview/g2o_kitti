import numpy as np
import cv2
from skimage.measure import ransac

def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

# def fundamentalToRt(F):
#     W = np.mat([[0,-1,0],[1,0,0],[0,0,1]],dtype=float)
#     U,d,Vt = np.linalg.svd(F)
#     if np.linalg.det(U) < 0:
#         U *= -1.0
#     if np.linalg.det(Vt) < 0:
#         Vt *= -1.0
#     R = np.dot(np.dot(U, W), Vt)
#     if np.sum(R.diagonal()) < 0:
#         R = np.dot(np.dot(U, W.T), Vt)
#     t = U[:, 2]

#     # TODO: Resolve ambiguities in better ways. This is wrong.
#     if t[2] < 0:
#         t *= -1
#     # return np.linalg.inv(poseRt(R, t))
#     return poseRt(R, t)


# class EssentialMatrixTransform(object):
#     def __init__(self):
#         self.params = np.eye(3)

#     def __call__(self, coords):
#         coords_homogeneous = np.column_stack([coords, np.ones(coords.shape[0])])
#         return coords_homogeneous @ self.params.T

#     def estimate(self, src, dst):
#         # assert src.shape == dst.shape
#         # assert src.shape[0] >= 8

#         # Setup homogeneous linear equation as dst' * F * src = 0.
#         A = np.ones((src.shape[0], 9))
#         A[:, :2] = src
#         A[:, :3] *= dst[:, 0, np.newaxis]
#         A[:, 3:5] = src
#         A[:, 3:6] *= dst[:, 1, np.newaxis]
#         A[:, 6:8] = src

#         # Solve for the nullspace of the constraint matrix.
#         _, _, V = np.linalg.svd(A)
#         F = V[-1, :].reshape(3, 3)

#         # Enforcing the internal constraint that two singular values must be
#         # non-zero and one must be zero.
#         U, S, V = np.linalg.svd(F)
#         S[0] = S[1] = (S[0] + S[1]) / 2.0
#         S[2] = 0
#         self.params = U @ np.diag(S) @ V

#         return True
#     def residuals(self, src, dst):
#         # Compute the Sampson distance.
#         src_homogeneous = np.column_stack([src, np.ones(src.shape[0])])
#         dst_homogeneous = np.column_stack([dst, np.ones(dst.shape[0])])

#         F_src = self.params @ src_homogeneous.T
#         Ft_dst = self.params.T @ dst_homogeneous.T

#         dst_F_src = np.sum(dst_homogeneous * F_src.T, axis=1)

#         return np.abs(dst_F_src) / np.sqrt(F_src[0] ** 2 + F_src[1] ** 2
#                                             + Ft_dst[0] ** 2 + Ft_dst[1] ** 2)


class VO_frontend(object):
    def __init__(self):
        self.detector = cv2.GFTTDetector_create(     #this has detetctors i.e kps, corresponding descriptors, kp matcher
                maxCorners=1000, minDistance=15.0, 
                qualityLevel=0.001, useHarrisDetector=False)
        self.descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(
                bytes=32, use_orientation=False)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    def get_keypoints(self, image):
        kps = self.detector.detect(image)
        kps, des = self.descriptor.compute(image, kps)
        # orb = cv2.ORB_create()
        # pts = cv2.goodFeaturesToTrack(image.astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
        # kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
        # kps, des = orb.compute(image, kps)
        return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des
    def get_matches(self, image1, image2):
        idx1 = []
        idx2 = []
        matches = self.matcher.match(image1.des, image2.des)     #(N1, 32) and (N2, 32) and matches will be list of len max(N1, N2)
        # print("Number of matches : ", len(matches))
        matches = sorted(matches, key = lambda x:x.distance)
        for match in matches:
            idx1.append(match.queryIdx)
            idx2.append(match.trainIdx)
        return idx1, idx2, matches
    def get_pose(self, image1, idx1, image2, idx2, K):  #here we find relative pose b/w 2 images
        Rt = np.eye(4)
        E, mask = cv2.findEssentialMat(image1.kps[idx1], image2.kps[idx2], K, cv2.RANSAC)    #get E matrix from kps matches using RANSAC
        _, rot, trans, mask = cv2.recoverPose(E, image1.kps[idx1], image2.kps[idx2], K)   #get (R, t) from cv2.recoverPose (here we need E, K, matches kps)
        # first arg is 3d points, [inf is hardcoded as 50], any points beyond 50 will not be considered for Rt estimation (we need to change this)
        Rt[:3, :3] = rot
        Rt[:3, 3] = trans.squeeze()
        return np.linalg.inv(Rt), E    #inverse becoz we need pose transfromation from second most recent frame to most recent frame
    def get_pose_using_cotracker_correspondences(self, frame1, frame2, K, cotracker_correspondences, cotracker_kps, idx, correspondences_keys, kps_keys):   #here we find relative pose b/w 2 images
        Rt = np.eye(4)
        img1_pixel_coordinates = cotracker_correspondences[correspondences_keys[idx - 1]][0]   #idx -1 as we skipped first frame (no frame before that)
        img2_pixel_coordinates = cotracker_correspondences[correspondences_keys[idx - 1]][1]
        # img1_pixel_coordinates = cotracker_kps[kps_keys[idx - 1]]   
        # img2_pixel_coordinates = cotracker_kps[kps_keys[idx]]
        E, mask = cv2.findEssentialMat(img1_pixel_coordinates, img2_pixel_coordinates, K, cv2.RANSAC)    #get E matrix from kps matches using RANSAC
        traingulated_pnts, rot, trans, mask = cv2.recoverPose(E, img1_pixel_coordinates, img2_pixel_coordinates, K)   #get (R, t) from cv2.recoverPose (here we need E, K, matches kps)
        # first arg is 3d points, [inf is hardcoded as 50], any points beyond 50 will not be considered for Rt estimation (we need to change this)
        Rt[:3, :3] = rot
        Rt[:3, 3] = trans.squeeze()
        return np.linalg.inv(Rt), E    #inverse becoz we need pose transfromation from second most recent frame to most recent frame
    
    def add_ones(self, x):
        if len(x.shape) == 1:
            return np.concatenate([x,np.array([1.0])], axis=0)
        else:
            return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    def get_triangulation(self, image1, idx1, image2, idx2, cotracker_correspondences, idx, keys):        
        # kps1_normalized = np.dot(np.linalg.inv(image1.K), self.add_ones(image1.kps[idx1]).T)
        # # print(kps1_normalized)
        # # print(kps1_normalized[:2])
        # kps2_normalized = np.dot(np.linalg.inv(image2.K), self.add_ones(image2.kps[idx2]).T)
        img1_pixel_coordinates = cotracker_correspondences[keys[idx - 1]][0]
        img2_pixel_coordinates = cotracker_correspondences[keys[idx - 1]][1]
        kps1_normalized = np.dot(np.linalg.inv(image1.K), self.add_ones(img1_pixel_coordinates).T)
        kps2_normalized = np.dot(np.linalg.inv(image2.K), self.add_ones(img2_pixel_coordinates).T)    
        pts_4d = cv2.triangulatePoints(image1.pose[:3, :], image2.pose[:3, :],
                                       kps1_normalized[:2], kps2_normalized[:2])
        return pts_4d.T
    
    def get_triangulation_cotracker(self, image1, image2, cotracker_correspondences, cotracker_kps, idx, correspondences_keys, kps_keys):
        # kps1_normalized = np.dot(np.linalg.inv(image1.K), self.add_ones(image1.kps[idx1]).T)
        # # print(kps1_normalized)
        # # print(kps1_normalized[:2])
        # kps2_normalized = np.dot(np.linalg.inv(image2.K), self.add_ones(image2.kps[idx2]).T)
        img1_pixel_coordinates = cotracker_correspondences[correspondences_keys[idx - 1]][0]
        img2_pixel_coordinates = cotracker_correspondences[correspondences_keys[idx - 1]][1]
        kps1_normalized = np.dot(np.linalg.inv(image1.K), self.add_ones(img1_pixel_coordinates).T)
        kps2_normalized = np.dot(np.linalg.inv(image2.K), self.add_ones(img2_pixel_coordinates).T)    
        pts_4d = cv2.triangulatePoints(image1.pose[:3, :], image2.pose[:3, :],
                                       kps1_normalized[:2], kps2_normalized[:2])
        return pts_4d.T
    # def get_matches(self, image1, image2):
    #     ret = []
    #     idx1 = []
    #     idx2 = []
    #     matches = self.matcher.match(image1.des, image2.des)
    #     print("Number of matches : ", len(matches))
    #     matches = sorted(matches, key = lambda x:x.distance)
    #     for match in matches:
    #         idx1.append(match.queryIdx)
    #         idx2.append(match.trainIdx)
    #         ret.append((image1.kps[match.queryIdx], image2.kps[match.trainIdx]))
    #     # assert len(ret) >= 8
    #     ret = np.array(ret)
    #     idx1 = np.array(idx1)
    #     idx2 = np.array(idx2)
    #     model, inliers = ransac((ret[:, 0], ret[:, 1]),
    #                             EssentialMatrixTransform,
    #                             min_samples=8,
    #                             residual_threshold=0.02,
    #                             max_trials=100)
    #     print("Matches:  %d -> %d -> %d -> %d" % (len(image1.des), len(matches), len(inliers), sum(inliers)))   
    #     return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)
    # def get_matches(self, frame1, frame2):
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    #     matches = bf.knnMatch(frame1.des, frame2.des, k=2)
    #     # Lowe's ratio test
    #     ret = []
    #     idx1, idx2 = [], []
    #     idx1s, idx2s = set(), set()
    #     for m,n in matches:
    #         if m.distance < 0.75*n.distance:
    #             p1 = frame1.kps[m.queryIdx]
    #             p2 = frame2.kps[m.trainIdx]
    #         if m.distance < 32:
    #             if m.queryIdx not in idx1s and m.trainIdx not in idx2s:
    #                 idx1.append(m.queryIdx)
    #                 idx2.append(m.trainIdx)
    #                 idx1s.add(m.queryIdx)
    #                 idx2s.add(m.trainIdx)
    #                 ret.append((p1, p2))
    #     # no duplicates
    #     assert(len(set(idx1)) == len(idx1))
    #     assert(len(set(idx2)) == len(idx2))
    #     assert len(ret) >= 8
    #     ret = np.array(ret)
    #     idx1 = np.array(idx1)
    #     idx2 = np.array(idx2)

    #     # fit matrix
    #     model, inliers = ransac((ret[:, 0], ret[:, 1]),
    #                             EssentialMatrixTransform,
    #                             min_samples=8,
    #                             residual_threshold=0.02,
    #                             max_trials=100)
    #     print("Matches:  %d -> %d -> %d -> %d" % (len(frame1.des), len(matches), len(inliers), sum(inliers)))
    #     print(model.params)
    #     print(np.linalg.inv(fundamentalToRt(model.params)))
    #     return idx1[inliers], idx2[inliers], fundamentalToRt(model.params)