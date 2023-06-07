import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from modules.constant import POSE_MAP, HAND_MAP, FACE_MAP, return_scale_resoution
from cycler import cycler
import matplotlib as mpl


# Define a color cycle
color_cycler = cycler(color=mpl.cm.tab10.colors)

# Use the color cycle for the plot
plt.rc('axes', prop_cycle=color_cycler)


class KeypointRenderer:
    def __init__(self, include_face=False, include_hand=True, linewidth=5.0, mode="2d", height=200, width=200, fps=30):
        self.include_face = include_face
        self.include_hand = include_hand
        self.linewidth = linewidth
        self.mode = mode
        self.height = height
        self.width = width
        self.fps = fps
    
    def is_numpy(self, keypoint):
        if not isinstance(keypoint, np.ndarray):
            keypoint = keypoint.numpy()
        return keypoint
    
    def set_lengths(self, reference, generated):
        nframes, gen_nframes = reference.shape[0], generated.shape[0]
        if gen_nframes < nframes:
            pad_width = ((0, nframes - gen_nframes), (0, 0), (0, 0))
            generated = np.pad(generated, pad_width, mode='constant')
        elif gen_nframes > nframes:
            generated = generated[:nframes, :, :]
        return generated
    
    def split_keypoint(self, keypoint):
        pose_idx = len(POSE_MAP) + 1
        right_hand_idx = pose_idx + len(HAND_MAP) + 1
        left_hand_idx = right_hand_idx + len(HAND_MAP) + 1

        pose_keypoints = keypoint[:, :pose_idx]
        hand_left_keypoints = keypoint[:, pose_idx:right_hand_idx] 
        hand_right_keypoints = keypoint[:, right_hand_idx:left_hand_idx] 
        face_keypoints = keypoint[:, left_hand_idx:]
        
        return (pose_keypoints, hand_left_keypoints, hand_right_keypoints, face_keypoints)

    def save_2d_animation(self, reference, generated, text="sample_text", save_path=None):
        reference, generated = map(lambda x: self.is_numpy(x), [reference, generated])
        nframes = reference.shape[0]

        if reference.shape[-1] == 3:
            reference, generated = map(lambda x: x[..., :2], [reference, generated])
        
        generated = self.set_lengths(reference, generated)
        
        ref_splited, gen_splited = map(lambda x: self.split_keypoint(x), [reference, generated])
        
        ref_pose, ref_left, ref_right, ref_face = ref_splited
        gen_pose, gen_left, gen_right, gen_face = gen_splited
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Define a function to update the plot at each frame
        def update(frame):
            # clear ax1 and ax2
            ax1.clear()
            ax2.clear()

            ax1.axis('off')
            ax2.axis('off')

            # Set the limits of the plot to include all keypoints
            ax1.set_xlim(-int(self.width), int(self.width)) # for x axis
            ax1.set_ylim(-int(self.height), int(self.height)) # for y axis

            ax2.set_xlim(-int(self.width), int(self.width)) # for x axis
            ax2.set_ylim(-int(self.height), int(self.height)) # for y axis

            ax1.annotate("Ground-truth", xy=(0.5, 1), xycoords='axes fraction', ha='center', va='bottom', fontsize=20)
            ax2.annotate("Generated", xy=(0.5, 1), xycoords='axes fraction', ha='center', va='bottom', fontsize=20)

            # Add text in the middle bottom side of the screen
            fig.text(0.5, 0, text, ha='center', va='bottom', fontsize=20)

            # Set colors
            colors = mpl.cm.tab10.colors

            # plot the pose keypoints
            for i, (j1, j2) in enumerate(POSE_MAP):
                color = colors[i % len(colors)]
                
                x1, y1 = ref_pose[frame, j1]
                x2, y2 = ref_pose[frame, j2]
                ax1.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth)
                
                x1, y1 = gen_pose[frame, j1]
                x2, y2 = gen_pose[frame, j2]
                ax2.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth)
            
            if self.include_hand:
                # plot the left hand keypoints
                for i, (j1, j2) in enumerate(HAND_MAP):
                    color = colors[i % len(colors)]

                    x1, y1 = ref_left[frame, j1]
                    x2, y2 = ref_left[frame, j2]
                    ax1.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                    x1, y1 = gen_left[frame, j1]
                    x2, y2 = gen_left[frame, j2]
                    ax2.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                # plot the right hand keypoints
                for i, (j1, j2) in enumerate(HAND_MAP):
                    color = colors[i % len(colors)]

                    x1, y1 = ref_right[frame, j1]
                    x2, y2 = ref_right[frame, j2]
                    ax1.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                    x1, y1 = gen_right[frame, j1]
                    x2, y2 = gen_right[frame, j2]
                    ax2.plot([x1, x2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

            if self.include_face:
                # plot the face keypoints
                for j1, j2 in FACE_MAP:
                    x1, y1 = ref_face[frame, j1]
                    x2, y2 = ref_face[frame, j2]
                    ax1.plot([x1, x2], [-y1, -y2], color='black', linewidth=self.linewidth * 0.5)

                    x1, y1 = gen_face[frame, j1]
                    x2, y2 = gen_face[frame, j2]
                    ax2.plot([x1, x2], [-y1, -y2], color='black', linewidth=self.linewidth * 0.5)

            return ax1, ax2
        
        anim = FuncAnimation(fig, update, frames=nframes)
        anim.save(save_path, writer='Pillow', fps=self.fps)

    def save_3d_animation(self, reference, generated, text="sample_text", save_path=None):
        reference, generated = map(lambda x: self.is_numpy(x), [reference, generated])
        nframes = reference.shape[0]

        generated = self.set_lengths(reference, generated)   

        ref_splited, gen_splited = map(lambda x: self.split_keypoint(x), [reference, generated])
        
        ref_pose, ref_left, ref_right, ref_face = ref_splited
        gen_pose, gen_left, gen_right, gen_face = gen_splited   

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': '3d'})  

        ax1.view_init(elev=10, azim=70)
        ax2.view_init(elev=10, azim=70)

        # Define a function to update the plot at each frame
        def update(frame):
            # clear ax1 and ax2
            ax1.clear()
            ax2.clear()

            # Set the limits of the plot to include all keypoints
            ax1.set_xlim(-int(self.width), int(self.width)) # for x axis
            ax1.set_zlim(-int(self.height), int(self.height)) # for y axis
            ax1.set_ylim(-(int((self.width + self.height) * 0.5)), int((self.width + self.height) * 0.5)) # for z axis

            ax2.set_xlim(-int(self.width), int(self.width)) # for x axis
            ax2.set_zlim(-int(self.height), int(self.height)) # for y axis
            ax2.set_ylim(-(int((self.width + self.height) * 0.5)), int((self.width + self.height) * 0.5)) # for z axis

            ax1.annotate("Ground-truth", xy=(0.5, 1), xycoords='axes fraction', ha='center', va='bottom', fontsize=20)
            ax2.annotate("Generated", xy=(0.5, 1), xycoords='axes fraction', ha='center', va='bottom', fontsize=20)

            # Add text in the middle bottom side of the screen
            fig.text(0.5, 0, text, ha='center', va='bottom', fontsize=20)

            # Set colors
            colors = mpl.cm.tab10.colors

            # plot the pose keypoints
            for i, (j1, j2) in enumerate(POSE_MAP):
                color = colors[i % len(colors)]
                x1, y1, z1 = ref_pose[frame, j1]
                x2, y2, z2 = ref_pose[frame, j2]
                ax1.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth)

                x1, y1, z1 = gen_pose[frame, j1]
                x2, y2, z2 = gen_pose[frame, j2]
                ax2.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth)

            if self.include_hand:
                # plot the left hand keypoints
                for i, (j1, j2) in enumerate(HAND_MAP):
                    color = colors[i % len(colors)]
                    x1, y1, z1 = ref_left[frame, j1]
                    x2, y2, z2 = ref_left[frame, j2]
                    ax1.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                    x1, y1, z1 = gen_left[frame, j1]
                    x2, y2, z2 = gen_left[frame, j2]
                    ax2.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                # plot the right hand keypoints
                for i, (j1, j2) in enumerate(HAND_MAP):
                    color = colors[i % len(colors)]
                    x1, y1, z1 = ref_right[frame, j1]
                    x2, y2, z2 = ref_right[frame, j2]
                    ax1.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

                    x1, y1, z1 = gen_right[frame, j1]
                    x2, y2, z2 = gen_right[frame, j2]
                    ax2.plot([x1, x2], [z1, z2], [-y1, -y2], color=color, linewidth=self.linewidth * 0.5)

            if self.include_face:
                # plot the face keypoints
                for j1, j2 in FACE_MAP:
                    x1, y1, z1 = ref_face[frame, j1]
                    x2, y2, z2 = ref_face[frame, j2]
                    ax1.plot([x1, x2], [z1, z2], [-y1, -y2], color='black', inewidth=self.linewidth * 0.5)

                    x1, y1, z1 = gen_face[frame, j1]
                    x2, y2, z2 = gen_face[frame, j2]
                    ax2.plot([x1, x2], [z1, z2], [-y1, -y2], color='black', inewidth=self.linewidth * 0.5)

            return ax1, ax2
    
        anim = FuncAnimation(fig, update, frames=nframes)
        anim.save(save_path, writer='Pillow', fps=self.fps)
    
    def save_animation(self, **kwargs):
        if self.mode == "3d":
            self.save_3d_animation(**kwargs)
        elif self.mode == "2d":
            self.save_2d_animation(**kwargs)
        else:
            raise NotImplementedError



def save_2d_keypoint(keypoints, width, height, include_hand=True, include_face=False):
    assert keypoints.shape[-1] == 2, "Must be 2D."
    
    fig = plt.figure(facecolor='black')
    ax = fig.add_subplot(111)

    pose_keypoints = keypoints[:8, :] #[frame, 8, 2]
    hand_left_keypoints = keypoints[8:29, :] #[frame, 21, 2]
    hand_right_keypoints = keypoints[29:50, :] #[frame, 21, 2]
    face_keypoints = keypoints[50:, :]

    for j1, j2 in POSE_MAP:
        x1, y1 = pose_keypoints[j1]
        x2, y2 = pose_keypoints[j2]
        ax.plot([x1, x2], [-y1, -y2], color='blue')

    if include_hand:
        # plot the left hand keypoints
        for j1, j2 in HAND_MAP:
            x1, y1 = hand_left_keypoints[j1]
            x2, y2 = hand_left_keypoints[j2]
            ax.plot([x1, x2], [-y1, -y2], color='blue', linewidth=1.0)

        for j1, j2 in HAND_MAP:
            x1, y1 = hand_right_keypoints[j1]
            x2, y2 = hand_right_keypoints[j2]
            ax.plot([x1, x2], [-y1, -y2], color='blue', linewidth=1.0)

    if include_face:
        # plot the face keypoints
        for j1, j2 in FACE_MAP:
            x1, y1 = face_keypoints[j1]
            x2, y2 = face_keypoints[j2]
            ax.plot([x1, x2], [-y1, -y2], color='blue', linewidth=1.0)

    # Set the axes limits to match the width and height
    ax.axis('off')
    ax.set_xlim(-int(width), int(width)) # for x axis
    ax.set_ylim(-int(height), int(height)) # for y axis

    return fig
        
