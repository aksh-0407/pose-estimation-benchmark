import numpy as np
import time
import json
import os

# ==========================================
# CORE ENGINE: Real-Time State Machine
# ==========================================
class RealTimeBowlerTracker:
    def __init__(self, camera_matrices, bowler_lock_frame=20):
        """
        Initializes the runtime tracker.
        :param camera_matrices: Dict of 3x4 projection matrices.
        :param bowler_lock_frame: How many frames to observe before locking the bowler ID.
        """
        self.camera_matrices = camera_matrices
        self.frame_count = 0
        self.bowler_id = None
        self.player_y_history = {} 
        self.bowler_lock_frame = bowler_lock_frame

    def triangulate_n_views(self, points_2d, projection_matrices):
        """Standard SVD Triangulation for a single 3D point."""
        num_cams = len(points_2d)
        if num_cams < 2: return None
        
        A = np.zeros((num_cams * 2, 4))
        for i in range(num_cams):
            x, y = points_2d[i]
            P = projection_matrices[i]
            A[i * 2]     = x * P[2, :] - P[0, :]
            A[i * 2 + 1] = y * P[2, :] - P[1, :]
            
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        return X[:3].tolist()

    def process_live_frame(self, current_frame_data):
        """
        Ingests a single millisecond of broadcast data and returns 3D coordinates.
        """
        self.frame_count += 1
        
        # PHASE 5: If bowler is not locked, run the tracking classifier
        if self.bowler_id is None:
            self._track_and_classify_roles(current_frame_data)
            
            # NEW: If it just locked the bowler on this exact frame, don't wait!
            # Generate the skeleton immediately so we don't waste the frame.
            if self.bowler_id is not None and self.bowler_id in current_frame_data:
                return self._build_3d_skeleton(current_frame_data[self.bowler_id])
            return None 
            
        # PHASE 6: If bowler is already locked, output the 3D skeleton instantly
        if self.bowler_id in current_frame_data:
            return self._build_3d_skeleton(current_frame_data[self.bowler_id])
        else:
            return None

    def _track_and_classify_roles(self, frame_data):
        """Tracks ankles frame-by-frame to find the sprinting bowler."""
        for g_id, cam_data in frame_data.items():
            available_cams = list(cam_data.keys())
            if len(available_cams) < 2: continue
            
            root_points_2d = []
            proj_matrices = []
            
            for cam_id in available_cams:
                kpts = cam_data[cam_id]["keypoints"]
                # Midpoint of ankles
                root_x = (kpts[15][0] + kpts[16][0]) / 2.0
                root_y = (kpts[15][1] + kpts[16][1]) / 2.0
                
                root_points_2d.append([root_x, root_y])
                proj_matrices.append(self.camera_matrices[cam_id])
                
            root_3d = self.triangulate_n_views(root_points_2d, proj_matrices)
            
            if root_3d:
                if g_id not in self.player_y_history:
                    self.player_y_history[g_id] = []
                self.player_y_history[g_id].append(root_3d[1])
                
        # Lock logic
        if self.frame_count >= self.bowler_lock_frame:
            max_displacement = 0
            best_id = None
            
            for g_id, y_track in self.player_y_history.items():
                if len(y_track) > 0:
                    displacement = abs(max(y_track) - min(y_track))
                    if displacement > max_displacement:
                        max_displacement = displacement
                        best_id = g_id
                        
            self.bowler_id = best_id
            print(f"\n[SYSTEM LOCKED] Bowler identified as {self.bowler_id} at Frame {self.frame_count}")
            self.player_y_history.clear() # Free up RAM

    def _build_3d_skeleton(self, bowler_cam_data):
        """Triangulates all 17 joints for the Unreal Engine."""
        available_cameras = list(bowler_cam_data.keys())
        if len(available_cameras) < 2: return None
        
        skeleton_3d = []
        for joint_idx in range(17):
            joint_2d_points = []
            joint_proj_matrices = []
            
            for cam_id in available_cameras:
                keypoints = bowler_cam_data[cam_id]["keypoints"]
                joint_2d_points.append(keypoints[joint_idx])
                joint_proj_matrices.append(self.camera_matrices[cam_id])
                
            joint_3d = self.triangulate_n_views(joint_2d_points, joint_proj_matrices)
            skeleton_3d.append(joint_3d)
            
        return skeleton_3d


## ==========================================
# EXECUTION: Production Run (Unreal Engine Export)
# ==========================================
if __name__ == "__main__":
    import json
    import os
    import time
    
    # 1. Initialize YOUR REAL projection matrices here
    my_camera_matrices = {f"cam_{i:02d}": np.random.rand(3, 4) for i in range(1, 8)}
    
    # 2. Start Tracker (Locking at Frame 30 to ensure accurate run-up tracking)
    tracker = RealTimeBowlerTracker(my_camera_matrices, bowler_lock_frame=30)
    
    # 3. Open your real 600-frame files
    files_to_open = [
        "cam_01.jsonl", "cam_02.jsonl", "cam_03.jsonl", 
        "cam_04.jsonl", "cam_05.jsonl", "cam_06.jsonl", "cam_07.jsonl"
    ]
    file_handles = [open(f, 'r') for f in files_to_open if os.path.exists(f)]

    print("="*50)
    print(">>> STARTING PRODUCTION PIPELINE (600 FRAMES)")
    print("="*50)

    # This dictionary will store the final timeline for Unreal Engine
    unreal_engine_export = {}

    if len(file_handles) == 7:
        frame_counter = 0
        total_start_time = time.time()
        
        # 4. Read the broadcast feed line-by-line simultaneously
        while True:
            lines = [f.readline() for f in file_handles]
            
            if not all(lines): 
                break # All 600 lines finished!
                
            frame_counter += 1
            incoming_frame_data = {}
            
            for line in lines:
                if not line.strip(): continue
                cam_data = json.loads(line)
                cam_id = cam_data["camera_id"]
                
                for player in cam_data.get("players", []):
                    g_id = player["global_player_id"]
                    if g_id not in incoming_frame_data:
                        incoming_frame_data[g_id] = {}
                    incoming_frame_data[g_id][cam_id] = {"keypoints": player["pose_2d"]["keypoints_norm"]}
                    
            # 5. Process the Frame
            live_3d_skeleton = tracker.process_live_frame(incoming_frame_data)
            
            # 6. Store the Output Silently
            if live_3d_skeleton:
                # Save all 17 joints for this frame into our export dictionary
                unreal_engine_export[frame_counter] = live_3d_skeleton
                
                # Print a clean progress update every 50 frames
                if frame_counter % 50 == 0:
                    print(f"Processed Frame {frame_counter}/600...")
                    
        for f in file_handles:
            f.close()
            
        total_time = time.time() - total_start_time
        
        # 7. EXPORT TO UNREAL ENGINE JSON
        export_filename = "unreal_engine_bowler_skeleton.json"
        with open(export_filename, "w") as outfile:
            json.dump(unreal_engine_export, outfile, indent=4)
            
        print("="*50)
        print(f">>> PIPELINE COMPLETE IN {total_time:.2f} SECONDS")
        print(f">>> Full 17-point 3D timeline saved to: {export_filename}")
        print("="*50)
    else:
        print("[ERROR] Missing camera files. Check your folder.")