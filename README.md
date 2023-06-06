# relative_pose
`relative_pose.py` is a simple script doing the following:
1. Find a good focal length from 2D-3D correspondences provided in `vr2d.pny` and `vr3d.npy`. 
2. Extract SIFT features from images.
3. Match features with kNN. 
4. Estimate the essential matrix between two images using feature correspondences.
5. Recover rotation and (up-to-a-scale) translation from the essential matrix using the focal length calculated at step 1.

I also ran COLMAP on the images using the focal length estimated and defining shared intrinsics with no distortion. The reconstruction error was around 0.2 pixels after bundle adjustment. You can find a snapshot in `colmap/colmap_scene.png` or should be able to open `colmap/project.ini' with COLMAP GUI.  
