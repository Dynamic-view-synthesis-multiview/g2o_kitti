import numpy as np
from process_frame import VO_frontend
from g2o_optimizer import optimize

def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

class Point(object):
    def __init__(self, mapp, loc, color, tid=None):   #loc is 3d point,  #for every point , we have (pt, color, id)
        self.pt = np.array(loc)
        self.frames = []      #set of frames from which current point is visible
        self.idxs = []    #indices of corresponding kps in respective frames (so, self.frames and self.idxs have same len)
        self.color = np.copy(color)
        self.id = mapp.add_point(self)  # we add point to Map (by passing Point object) and assign id incrementally as we add points ...
    def add_observation(self, frame, idx):
        assert frame.pts[idx] is None
        assert frame not in self.frames
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
    def homogeneous(self):
        return add_ones(self.pt)

class Frame(object):
    def __init__(self, mapp, image, K, cotracker_correspondences, cotracker_kps, idx, correspondences_keys, kps_keys, pose=np.eye(4), tid=None):    #for every frame , we have (img, K, pose, frontend, (kps, desc) from frontend obj, pts, id)
        self.image = image
        self.K = K
        self.pose = pose
        self.frontend = VO_frontend()
        # self.kps, self.des = self.frontend.get_keypoints(image)   #(kps :  (N, 2), des: (N, 32))
        # self.kps = cotracker_correspondences[correspondences_keys[idx]][0]    #have to fix this (bug)
        self.kps = cotracker_kps[kps_keys[idx]]
        # img2_pixel_coordinates = cotracker_correspondences[keys[idx - 1]][1]
        self.pts = [None]*len(self.kps)   #list of all None of len = N (same as self.kps)
        self.id = mapp.add_frame(self)    # we add current frame to Map (by passing Frame object) and assign id incrementally as camera captures scene
        self.SE3 = None   #this is g2o SE3 instance
    @property
    def normalized_kps(self):
        if not hasattr(self, '_normalized_kps'):
            self._normalized_kps = np.dot(np.linalg.inv(self.K), add_ones(self.kps).T).T[:, :2]
        return self._normalized_kps

class Map(object):
    def __init__(self):
        self.frames = []   # has collection of Frame objects
        self.points = []   # has collection of Point objects
        self.max_frame = 0    #no of frames encountered so far
        self.max_point = 0   ##no of points added so far
    def add_point(self, point):
        ret = self.max_point
        self.max_point = self.max_point + 1
        self.points.append(point)
        return ret
    def add_frame(self, frame):
        ret = self.max_frame
        self.max_frame = self.max_frame + 1
        self.frames.append(frame)
        return ret
    def optimize(self, local_window=20, fix_points=False, verbose=False, iterations=50):
        error = optimize(self.frames, self.points, local_window, fix_points, verbose, iterations)        #main logic of g2o optimization
        # print(error)
        return error
        # culled_pt_count = 0
        # for p in self.points:
        #     old_point = len(p.frames)
    def test(self):
        for f in self.frames[-1:]:
            print(f)
        # print(self.frames[:])
