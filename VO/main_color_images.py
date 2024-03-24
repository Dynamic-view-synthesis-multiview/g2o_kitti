import numpy as np
import cv2
from mapp import Frame, Map, Point
import glob
from process_frame import VO_frontend
from display import Display3D


def process_frame(image, pose=None, verts=None):    #Note: We consider frame of first frame as world frame when starting VO
    frame = Frame(mapp, image, K)   #mapp is Map object which has frames (Frame instances) and points as attributes, we also add current Frame instance in Map i.e mapp
    print()
    print("******** Frame %d *********" % frame.id)
    # print("Keypoiints : ")
    # print(frame.kps)
    # print(frame.normalized_kps)
    if frame.id == 0:
        return 0, 0, 0
    
    # Consider latest 2 frames
    frame1 = mapp.frames[-1]  #most recent frame
    frame2 = mapp.frames[-2]  #second most recent frame  --> we need pose from (second recent to most recent)
    
    # idx1, idx2, Rt = frontend.get_matches(frame1, frame2)
    idx1, idx2, matches = frontend.get_matches(frame1, frame2)     #lets say len(matches) = N ==> N= max(N1 -- no of desc in img1, N2 -- no of desc in img2) and len(idx1) = len(idx2) = N, idx1 and idx2 are indices over kps in respective frames 
    Rt, E = frontend.get_pose(frame1, idx1, frame2, idx2, K)

    # for i,idx in enumerate(idx2):
    #     if frame2.pts[idx] is not None and frame1.pts[idx1[i]] is None:
    #         frame2.pts[idx].add_observation(frame1, idx1[i])
    frame1.pose = np.dot(Rt, frame2.pose)   #get pose of most recent frame from [pose of its prev frame and rel transfromation i.e Rt]
    
    pts_4d = frontend.get_triangulation(frame1, idx1, frame2, idx2)   #(N, 4) where N is no of matches b/w recent 2 frames 
    pts_3d = pts_4d[:, :4] / pts_4d[:, 3:]   #(N, 4) but last col is entirely 1.0
    
    new_pts_count = 0   # filtering out 3D points based on certain criteria after the triangulation
    for i, p in enumerate(pts_3d):
        bool_pts1 = False
        bool_pts2 = False
        
        # Transform 3D point into camera coordinates for both frames
        pl1 = np.dot(frame1.pose, p)   #(4, ) 
        pl2 = np.dot(frame2.pose, p)
        # print(pl1, pl2)
        
        # Check if the point is in front of both cameras
        if pl1[2] < 0 or pl2[2] < 0:
            continue
        
        # Project 3D points onto image planes and calculate reprojection errors
        pp1 = np.dot(frame1.K, pl1[:3])  
        pp2 = np.dot(frame2.K, pl2[:3])
        pp1 = (pp1[0:2] / pp1[2]) - frame1.kps[idx1[i]]
        pp2 = (pp2[0:2] / pp2[2]) - frame2.kps[idx2[i]]
        pp1 = np.sum(pp1**2)
        pp2 = np.sum(pp2**2)
        # print(pp1, pp2)
        
        # Check if the reprojection errors meet certain criteria
        if pp1 > 2 or pp2 > 2:
            continue
        try:
            color = img[int(round(frame1.kps[idx1[i],1])), int(round(frame1.kps[idx1[i],0]))]    #doubt:  whats this img (current img) ?
        except IndexError:
            color = (255,0,0)
        pt = Point(mapp, p[0:3], color)   #create Point obj in Map with corresponding (XYZ) and color
        if frame2.pts[idx2[i]] is None:    #Note that current point (didnt get filtered) will now be visible from frame 1 and frame2
            pt.add_observation(frame2, idx2[i])
            bool_pts2 = True
        if frame1.pts[idx1[i]] is None:
            pt.add_observation(frame1, idx1[i])
            bool_pts1 = True
            
        # If both bool_pts1 and bool_pts2 are True, increment the new_pts_count
        if bool_pts1 and bool_pts2:
            new_pts_count += 1

    print("Adding:   %d new points" % (new_pts_count))
    if frame.id >= 4 and frame.id%5 == 0:
        # print("Optimizer")
        err = mapp.optimize(iterations=50)      #g2o optimization happens here (once every 5 frames): this is very similar to sliding window method in local BA
        print("Optimize: %f units of error" % err)
    print("Map:      %d points, %d frames" % (len(mapp.points), len(mapp.frames)))
    return frame1.pose[:3, 3]       #finally after entire optimization, return pose of last frame wrt world frame/0th frame which is I (identity) matrix


mapp = Map()
frontend = VO_frontend()
img_paths = sorted(glob.glob("/home2/jayaram.reddy/research_threads/dynamic_nerf_reconstruction/VO_pipeline/kitti/ims/0/*"))
W = int(1242.0)
H = int(375.0)
K = np.array([[718.8560,0,621.0],[0,718.8560,187.5],[0,0,1]])
disp3d = Display3D()

for path in img_paths:  #144 images for kitti
    img = cv2.imread(path)    #read only grayscale img: (375, 1242)
    x, y, z = process_frame(img)
    if disp3d is not None:
        disp3d.paint(mapp)