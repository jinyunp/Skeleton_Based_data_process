# 2023.05.11 박진윤 선임연구원, 태안군 인공지능유합산업진흥원
# 3D skeleton rotation, normalization 함수화

import numpy as np
import math
import sys
eps = sys.float_info.epsilon

# scale normalization
def ScaleNorm(datasc, joints, head_idx):
    norm_seg = []
    for segment in datasc:
        norm = []
        for frame in segment:
            z_val = []
            for i in range(2, joints*3, 3):
                z_val.append(frame[i])
            min_val = np.array(z_val).min()

            #scale normed value for each frame
            max_val = frame[(head_idx*3)+2]
            frame = frame/max_val
            norm.append(np.array(frame)) 
        #scale normed value for each segment
        norm_seg.append(np.array(norm))

    return np.array(norm_seg)   

# 좌표계 normalization
def AxisNorm(datasc, joints, spine_idx):
    zeroscale_seg = []
    for segment in datasc:
        zeroscale=[]
        for frame in segment:
            spine = frame[spine_idx*3:(spine_idx*3)+3]

            zero = []
            for i in range(0,joints*3,3):
                zero.append(frame[i] - spine[0])
                zero.append(frame[i+1] - spine[1])
                zero.append(frame[i+2] - spine[2])
            zeroscale.append(np.array(zero))
        zeroscale_seg.append(np.array(zeroscale))
        
    return np.array(zeroscale_seg)

#turning the skeleton facing positive y-axis
def ViewRot(skeleton, joints, right_hip_idx):
    from scipy.spatial.transform import Rotation as R
    sample_rot_seg = []
    for segment in skeleton:
        
        v1 = (segment[0][right_hip_idx*3],segment[0][right_hip_idx*3+1],0)
        v2 = (1,0,0)
        angle = angle_between(v1, v2)

        if (segment[0][right_hip_idx*3]>0)&(segment[0][right_hip_idx*3+1]>0):
            rotation_degrees = 360-np.degrees(angle)
        elif (segment[0][right_hip_idx*3]<0)&(segment[0][right_hip_idx*3+1]>0):
            rotation_degrees = 360-np.degrees(angle)
        else:
            rotation_degrees = np.degrees(angle)

        rotation_radians = np.radians(rotation_degrees)
        rotation_axis = np.array([0, 0, 1])
        rotation_vector = rotation_radians * rotation_axis
        rotation = R.from_rotvec(rotation_vector)
        
        sample_rot = []
        for frame_idx in range(len(segment)): #each line in segment

            rot_data = []

            # quaternion of all joints according to the core vector
            for i in range(joints):
                vec = segment[frame_idx][i*3:(i*3)+3]
                vec = (vec[0], vec[1], 0)

                rot_vec = rotation.apply(vec)

                x = float(rot_vec[0])
                y = float(rot_vec[1])
                z = segment[frame_idx][(i*3)+2]

                rot_data.append(x)
                rot_data.append(y)
                rot_data.append(z)

            sample_rot.append(np.array(rot_data))
            
        sample_rot_seg.append(np.array(sample_rot))

    return np.asarray(sample_rot_seg)


# https://stackoverflow.com/a/13849249
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u =( v1 / np.linalg.norm(v1)) + eps
    v2_u = (v2 / np.linalg.norm(v2)) +eps
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def augmentation_zaxis(skeleton, angle, joints):

    vrot_data = [] # rotation according to the vector

    # rotation
    for frame in skeleton: # rotation for each frame
        vrot = []
        z_axis = np.asarray([0,0,1])
        
        for i in range(joints): # for each keypoint in one frame
            vec = frame[i*3:(i*3)+3]
            vec = np.array([vec[0], vec[1], vec[2]])
            
            theta = (angle/10)*np.pi
            rot = (np.dot(rotation_matrix(z_axis, theta), vec))
            vrot.append(rot[0])
            vrot.append(rot[1])
            vrot.append(rot[2])# append each joint for one frame
            
        if np.isnan(vrot).any() == True:
                vrot = [0] * joints*3 
        vrot_data.append(vrot) #append each frame data
        
    vrot_data = np.asarray(vrot_data)

    return vrot_data

def augmentation_spine(skeleton, angle, joints, mid_hip, neck):

    vrot_data = [] # rotation according to the vector

    # rotation
    for frame in skeleton: # rotation for each frame
        vrot = []
        mid_hip_vec = frame[mid_hip*3:mid_hip*3+3]
        neck_vec = frame[neck*3:neck*3+3]
        spine_vector = np.asarray(neck_vec-mid_hip_vec)
        
        for i in range(joints): # for each keypoint in one frame
            vec = frame[i*3:(i*3)+3]
            vec = np.array([vec[0], vec[1], vec[2]])
            
            theta = (angle/10)*np.pi
            rot = (np.dot(rotation_matrix(spine_vector, theta), vec))
            vrot.append(rot[0])
            vrot.append(rot[1])
            vrot.append(rot[2])# append each joint for one frame
            
        if np.isnan(vrot).any() == True:
                vrot = [0] * joints*3 
        vrot_data.append(vrot) #append each frame data
        
    vrot_data = np.asarray(vrot_data)

    return vrot_data

####################################################################################
# SpineJointTranslationLayer(joints, sping_joint_index)
# SkeletonNormalizationLayer(joints)
# SkeletonRotationLayer(mid_hip_idx, neck_idx, num_joints, angle_list)

class SpineJointTranslationLayer:
    def __init__(self, joints, spine_joint_index):
        self.joints = joints
        self.spine_joint_index = spine_joint_index
        
    def translate_sequence(self, sequence):
        translated_sequence = []
        
#         spine_joint_coordinates = sequence[0][self.spine_joint_index*3:self.spine_joint_index*3+3]
        
        for frame in sequence:
            # Check if the sequence contains only zeros
            if np.all(frame == 0):
                translated_sequence.append(np.zeros(frame.shape))
            else:
                translated_frame = self.translate_frame(frame)
                translated_sequence.append(translated_frame)
            
        return np.array(translated_sequence)

    def translate_frame(self, inputs):
        spine_joint_coordinates = inputs[self.spine_joint_index*3:self.spine_joint_index*3+3]
        inputs = np.reshape(inputs, (self.joints, 3))
        translated_frame = inputs - spine_joint_coordinates
        translated_frame = np.reshape(translated_frame, (self.joints * 3,))

        return translated_frame
    
class SkeletonNormalizationLayer:
    def __init__(self, joints):
        self.joints = joints
    
    def nomalize_sequence(self, sequence):
        nomalized_sequence = []
        
        ## get data from first frame
        first_frame = np.reshape(sequence[0], (self.joints, 3))
        min_vals = np.min(first_frame, axis=0)[2]
        max_vals = np.max(first_frame, axis=0)[2]
        range_vals = max_vals - min_vals
        
        for frame in sequence:
            # Check if the sequence contains only zeros
            if np.all(frame == 0):
                nomalized_sequence.append(np.zeros(frame.shape))
            else:
                normalized_frame = self.normalize_frame(frame, range_vals)
                nomalized_sequence.append(normalized_frame)
            
        return np.array(nomalized_sequence)

    def normalize_frame(self, inputs, range_vals):
        inputs = np.reshape(inputs, (self.joints, 3))

        normalized_frame = inputs * (2/range_vals)
#         normalized_frame = 2 * (inputs - min_vals) / (max_vals - min_vals) - 1
        normalized_frame = np.reshape(normalized_frame, (self.joints * 3,))

        return normalized_frame

    
class SkeletonFacingYAxisLayer:
    def __init__(self, joints, left_hip_joint_index, right_hip_joint_index):
        self.joints = joints
        self.left_hip_joint_index = left_hip_joint_index
        self.right_hip_joint_index = right_hip_joint_index

    # Check if the sequence contains only zeros
    def yaxis_rotate_sequence(self, inputs):
        yaxis_rotated_sequence = []
        
        left_hip_coordinates = inputs[0][self.left_hip_joint_index*3:self.left_hip_joint_index*3+3]
        right_hip_coordinates = inputs[0][self.right_hip_joint_index*3:self.right_hip_joint_index*3+3]
        hip_vector = right_hip_coordinates-left_hip_coordinates

        # Normalize the hip_vector
        hip_vector_norm = np.linalg.norm(hip_vector)
        hip_vector_normalized = hip_vector / hip_vector_norm

        # Calculate the rotation angle
        angle = np.arccos(np.dot(hip_vector_normalized, (1,0,0)))

        if (hip_vector[0]>0)&(hip_vector[1]>0):
            angle = 2*np.pi - angle
        elif (hip_vector[0]<0)&(hip_vector[1]>0):
            angle = 2*np.pi - angle
        else:
            pass
        
        for frame in inputs:
            if np.all(inputs == 0):
                yaxis_rotated_sequence.append(np.zeros(frame.shape))
            else:
                rotated_frame = self.yaxis_rotate_frame(frame, angle)
                yaxis_rotated_sequence.append(rotated_frame)

        return np.array(yaxis_rotated_sequence)

    def yaxis_rotate_frame(self, frame, angle):

        frame = np.reshape(frame, (self.joints, 3))

        radian_angle = -angle

        rotation_matrix = np.array([[np.cos(radian_angle), -np.sin(radian_angle), 0],
                                    [np.sin(radian_angle), np.cos(radian_angle), 0],
                                    [0, 0, 1]])

        rotated_frame = np.dot(frame, rotation_matrix)
        rotated_frame = np.reshape(rotated_frame, (self.joints*3,))

        return rotated_frame
    
class SkeletonAugmentationLayer:
    def __init__(self, joints, angle_list):
#         self.num_augmentations = num_augmentations
        self.angle_list = angle_list
        self.joints = joints
        
    def augment_sequence(self, sequence):
        augmented_sequences = []
        
        for angle in self.angle_list:
            rotated_sequence = self.rotate_sequence(sequence, angle)
            augmented_sequences.append(rotated_sequence)
        
        return np.array(augmented_sequences)
    
    def rotate_sequence(self, sequence, angle):
        rotated_sequence = []
        
        for frame in sequence:
            rotated_frame = self.rotate_frame(frame, angle)
            rotated_sequence.append(rotated_frame)
        
        return np.array(rotated_sequence)
    
    def rotate_frame(self, frame, angle):
        # reshaping frame to shape (num_joints, 3) representing 3D coordinates of joints
        frame = np.reshape(frame, (self.joints,3))
        
        # Generate a random rotation angle for each augmentation
#         angle = np.random.uniform(0, angle)
        
        # Convert the angle to radians
        radian_angle = (angle/10)*np.pi
        
        # Perform rotation about the z-axis
        rotation_matrix = np.array([[np.cos(radian_angle), -np.sin(radian_angle), 0],
                                    [np.sin(radian_angle), np.cos(radian_angle), 0],
                                    [0, 0, 1]])
        
        rotated_frame = np.dot(frame, rotation_matrix)
        rotated_frame = np.reshape(rotated_frame, (self.joints*3,))
        
        return rotated_frame


class SkeletonRotationLayer:
    def __init__(self, mid_hip_idx, neck_idx, num_joints, angle_list):
        self.mid_hip_idx = mid_hip_idx
        self.neck_idx = neck_idx
        self.num_joints = num_joints
        self.angle_list = angle_list
        
    def spine_augment_sequence(self, sequence):
        augmented_sequence = []
#         rotation_matrix, mid_hip = self.calculate_rotation(sequence[0])
        
        for angle in self.angle_list:
            rotated_sequence = self.rotate_sequence(sequence, angle)
            augmented_sequence.append(rotated_sequence)
        
        return np.array(augmented_sequence)
    
    def rotate_sequence(self, sequence, angle):
        rotated_sequence = []
        rotation_matrix = self.calculate_rotation(sequence[0], angle)
        mid_hip = sequence[0][self.mid_hip_idx*3:self.mid_hip_idx*3+3]
        
        for frame in sequence:
            rotated_frame = self.rotate_frame(frame, rotation_matrix, mid_hip)
            rotated_sequence.append(rotated_frame)
            
        return np.array(rotated_sequence)
    
    def calculate_rotation(self, frame, angle):
        frame = np.reshape(frame, (self.num_joints, 3))
        
        # find the neck position
        neck = frame[self.neck_idx]
        mid_hip = frame[self.mid_hip_idx]
        
        # find the vector from mid hip to neck
        rotation_axis = neck - mid_hip
        
        # normalize the rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # create a rotation matrix using the axis-angle formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        return rotation_matrix
    
    def rotate_frame(self, frame, rotation_matrix, mid_hip):
        frame = np.reshape(frame, (self.num_joints, 3))
        
        # apply the rotation to each joint
#         rotated_frame = frame @ rotation_matrix.T
        rotated_frame = (frame - mid_hip) @ rotation_matrix.T + mid_hip
        rotated_frame = np.reshape(rotated_frame, (self.num_joints*3,))
        
        return rotated_frame


