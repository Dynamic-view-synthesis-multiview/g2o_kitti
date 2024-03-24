import g2o
import numpy as np

def poseRt(R, t):  #get 4*4 transformation matrix from [R, t]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret

def optimize(frames, points, local_window, fix_points, verbose=False, rounds=50):
    # print("Entering Optimizer")
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]     #last 20 frames in sliding window

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    # add normalized camera
    cam = g2o.CameraParameters(1.0, (0.0, 0.0), 0)       #focal length of 1.0, principal point at (0.0, 0.0), and no distortion (0).
    # cam = g2o.CameraParameters(718.8560, (607.1928, 185.2157), 0.0)         
    cam.set_id(0)                       
    opt.add_parameter(cam)   

    robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    
    # these are g2o vertices
    graph_frames, graph_points = {}, {}     #build g2o graph where there are 2 types of vertices (on for frame poses and other for points)

    # add frames to graph
    for f in (local_frames if fix_points else frames):
        pose = f.pose    #(4, 4) array
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])     #create SE3 g2o instance for pose vertex
        f.SE3 = se3
        v_se3 = g2o.VertexSE3Expmap()   #SE3 g2o vertex in g2o graph
        v_se3.set_estimate(se3)

        v_se3.set_id(f.id * 2)   # in g2o graph, pose vertices will have even indices and point vertices will have odd indices
        v_se3.set_fixed(f.id <= 1 or f not in local_frames)
        #v_se3.set_fixed(f.id != 0)
        opt.add_vertex(v_se3)    #add pose vertex to optimizer

        # confirm pose correctness
        est = v_se3.estimate()
        assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        assert np.allclose(pose[0:3, 3], est.translation())

        graph_frames[f] = v_se3

    # add points to g2o graph (for every point added in graph, we create an edge b/w point vertex and frame vertices of g2o graph)
    for p in points:
        if not any([f in local_frames for f in p.frames]):
            continue

        pt = g2o.VertexSBAPointXYZ()    # create point g2o instance for point vertex in g2o graph
        pt.set_id(p.id * 2 + 1)    # point vertices in g2o graph will have odd indices
        pt.set_estimate(p.pt[0:3])   # p.pt has shape = (3, )
        pt.set_marginalized(True)
        pt.set_fixed(fix_points)
        opt.add_vertex(pt)   # add point vertex to optimizer
        graph_points[p] = pt

        # add edges for each frame associated with current point p
        for f, idx in zip(p.frames, p.idxs):    #p.frames will have info of all frames from which point p is visible and p.idxs are corresponding kps indices in the frame/img, so len(p.frames) = len(p.idxs)
            if f not in graph_frames:
                continue
            edge = g2o.EdgeProjectXYZ2UV()
            edge.set_parameter_id(0, 0)
            edge.set_vertex(0, pt)
            edge.set_vertex(1, graph_frames[f])
            edge.set_measurement(f.normalized_kps[idx])   #weight of edge is the normalized kp co-ordinate/position in image
            edge.set_information(np.eye(2))
            edge.set_robust_kernel(robust_kernel)
            opt.add_edge(edge)    # add edge in g2o optimizer
            
            # print(cam.cam_map(f.SE3 * p.pt), f.normalized_kps[idx])
    # print('num vertices:', len(opt.vertices()))
    # print('num edges:', len(opt.edges()))    
    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(rounds)    #optimize g2o graph for 50 iterations  -- meaning graph_frames and graph_points are also optimized

    # put frames back -- the optimized camera poses and 3D point positions are extracted from the g2o optimization graph and updated back to the corresponding data structures.
    for f in graph_frames:
        est = graph_frames[f].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        f.pose = poseRt(R, t)

    # put points back
    if not fix_points:
        for p in graph_points:
            p.pt = np.array(graph_points[p].estimate())

    return opt.active_chi2()