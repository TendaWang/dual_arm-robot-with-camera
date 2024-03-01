import os
import math
import numpy as np
import time
import pybullet as p
import random
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

ROBOT_URDF_PATH = "./urdf/dual_arm_simple.urdf"
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
OBJ_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "duck_vhacd.urdf")


class D_Robot():

    def __init__(self, camera_attached=False):
        p.connect(p.GUI)
        p.setGravity(0,0,-9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.d_robot = self.load_robot()

        self.num_joints = p.getNumJoints(self.d_robot)  # 关节数

        self.left_control_joints = [ "l_joint1", "l_joint2", "l_joint3", "l_joint4", "l_joint5","l_joint6"]
        self.right_control_joints =[ "r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6"]
        self.head_control_joints=['joint_head_link_roll','joint_head_link_pitch']
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo",
                                     ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                      "controllable"])
        self.dof=6
        self.joints = AttrDict()
        self.add_gui_sliders()
        for i in range(self.num_joints):
            info = p.getJointInfo(self.d_robot, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.left_control_joints+self.right_control_joints+self.head_control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                                   jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                p.setJointMotorControl2(self.d_robot, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=100)
            self.joints[info.name] = info

        self.load_obj()
    def load_robot(self):
        flags = p.URDF_USE_SELF_COLLISION
        self.table = p.loadURDF(TABLE_URDF_PATH, [1.1, 0, 0], [0, 0, 0, 1],useFixedBase=True)
        self.table2 = p.loadURDF(TABLE_URDF_PATH, [0,1.1,  0], [0, 0, 0, 1],useFixedBase=True)
        self.table4 = p.loadURDF(TABLE_URDF_PATH, [0,-1.1,  0], [0, 0, 0, 1],useFixedBase=True)
        self.robot = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0.2], [0, 0, 0, 1], flags=flags,useFixedBase=True)
        self.planeId = p.loadURDF("plane.urdf")
        return self.robot
    def load_obj(self):
        self.obj = p.loadURDF(OBJ_URDF_PATH, basePosition=[0.4, 0, 0.6],useFixedBase=False)  # 加载一个物品
        return self.obj
    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        control_joints=self.left_control_joints+self.right_control_joints+self.head_control_joints

        for i, name in enumerate(control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        p.setJointMotorControlArray(
            self.d_robot, indexes,
            p.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0] * len(poses),
            positionGains=[0.04] * len(poses), forces=forces
        )

    def get_joint_angles(self):
        j = p.getJointStates(self.d_robot, [1, 2, 3, 4, 5, 6])
        joints = [i[0] for i in j]
        return joints

    def check_collisions(self):
        collisions = p.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False

    def calculate_ik(self, position, orientation):
        orientation=[p.getQuaternionFromEuler(i) for i in orientation]

        joint_angles = p.calculateInverseKinematics(
            self.d_robot, 6, position[0], orientation[0],
        )
        joint_angles2 = p.calculateInverseKinematics(
            self.d_robot, 13, position[1], orientation[1],
        )
        joint_angles=list(joint_angles)[:6]+list(joint_angles2)[6:]

        return joint_angles

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(p.addUserDebugParameter("XL", -1, 1, 0))
        self.sliders.append(p.addUserDebugParameter("YL", -1, 1, 0))
        self.sliders.append(p.addUserDebugParameter("ZL", -2, 2, 0))
        self.sliders.append(p.addUserDebugParameter("XR", -1, 1, 0))
        self.sliders.append(p.addUserDebugParameter("YR", -1, 1, 0))
        self.sliders.append(p.addUserDebugParameter("ZR", -2, 2, 0))
        self.sliders.append(p.addUserDebugParameter("r", -4, 4, 0))
        self.sliders.append(p.addUserDebugParameter("p", -4, 4, 0))
        self.sliders.append(p.addUserDebugParameter("y", -4, 4, 0))
        self.sliders.append(p.addUserDebugParameter("head_r", -4, 4, 0))
        self.sliders.append(p.addUserDebugParameter("head_p", -4, 4, 0))



    def read_gui_sliders(self):  # 在这里放入位置参数
        xl = p.readUserDebugParameter(self.sliders[0])
        yl = p.readUserDebugParameter(self.sliders[1])
        zl = p.readUserDebugParameter(self.sliders[2])
        xr = p.readUserDebugParameter(self.sliders[3])
        yr = p.readUserDebugParameter(self.sliders[4])
        zr = p.readUserDebugParameter(self.sliders[5])
        r = p.readUserDebugParameter(self.sliders[6])
        pp = p.readUserDebugParameter(self.sliders[7])
        y = p.readUserDebugParameter(self.sliders[8])
        head_r = p.readUserDebugParameter(self.sliders[9])
        head_p = p.readUserDebugParameter(self.sliders[10])
        return [xl,yl,zl,xr,yr,zr,r,pp,y,head_r,head_p]

    def grasp_pos(self):
        left=        [0.4,0.27,0.99,3,-0.8,-1]
        right=[0.36,-0.25,0.99,-2.5,-3.4,-0.168]
        joints=self.calculate_ik([left[:3],right[0:3]],[left[3:6],right[3:6]])#desire_pos[6:9])
        joints[-1]=-1.25

        for i in range(10):
            self.set_joint_angles(joints)
            p.stepSimulation()
    def take_pic(self):

        width = 1080  # 图像宽度
        height = 720  # 图像高度
        fov = 50  # 相机视角
        aspect = width / height  # 宽高比
        near = 0.01  # 最近拍摄距离
        far = 20  # 最远拍摄距离
        #此参数与后续的建立三维点云相关，不得修改
        j3=self.joints['head_joint_camera'].id
        j2=self.joints['r_joint_camera'].id
        j1=self.joints['l_joint_camera'].id
        cameraLinkNum=[j1,j2,j3]#l,r,h

        depthimgs=[]
        rgbimgs=[]
        extrs=[]



        for i in range(3):

            state = p.getLinkState(self.robot, cameraLinkNum[i])
            cameraPos = np.array(state[0])

            qq=np.array(state[1])
            rot = R.from_quat(qq).as_matrix()

            targetPos = np.matmul(rot, np.array([0, 0,-1])) + cameraPos

            cameraupPos = np.matmul(rot, np.array([-1, 0, 0]))

            viewMatrix = p.computeViewMatrix(
                cameraEyePosition=cameraPos,
                cameraTargetPosition=targetPos,
                cameraUpVector=cameraupPos,
                physicsClientId=0
            )  # 计算视角矩阵
            viewMatrix=np.array(viewMatrix)
            vm=np.array(viewMatrix).reshape(4,4).T

            extrs.append(vm)  # camera extra parameters
            projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)  # 计算投影矩阵
            images = p.getCameraImage(width, height, viewMatrix, projection_matrix,
                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgbImg = cv2.cvtColor(images[2], cv2.COLOR_BGRA2BGR)

            depImg = far * near / (far - (far - near) * images[3])
            depImg = np.asanyarray(depImg).astype(np.float32) * 1000.
            depImg = (depImg.astype(np.uint16))
            rgbimgs.append(rgbImg)
            depthimgs.append(depImg)
            cv2.imshow('1',rgbImg)
            cv2.waitKey(0)

        return rgbimgs,depthimgs,extrs
    def pc_cam2pc_world(self,pc,extr):
        #left hand -> right hand
        depth=np.asarray(pc.points)
        for i in range(len(depth)):
            depth[i][0]=- depth[i][0]
        pc.points = o3d.utility.Vector3dVector(depth)
        R1 = extr[:3, :3].T
        T = extr[:3, 3]
        pc = pc.translate((T[0], T[1], T[2]), relative=True)
        pc = pc.rotate(R1, center=(0, 0, 0))

        return pc

    def make_3d(self,color_images,depth_images,extrs):
        clouds=[]
        for i in range(len(color_images)):
            color_image = o3d.geometry.Image(color_images[i])
            depth_image = o3d.geometry.Image(depth_images[i])
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image,
                                                                            convert_rgb_to_intensity=False)
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.Kinect2DepthCameraDefault)
            intrinsic.set_intrinsics(width=1080, height=720, fx=1.429 * 1080 / 2, fy=2.14450693 * 720 / 2, cx=1080 / 2,
                                     cy=720 / 2)
            # 投影矩阵如下：
            # (1.429671287536621,       0.0,                0.0,                    0.0,
            #  0.0,                     2.1445069313049316, 0.0,                    0.0,
            #  0.0,                     0.0,                -1.0010005235671997,    -1.0,
            #  0.0,                     0.0,                -0.020010003820061684,   0.0)
            point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)

            point_cloud=self.pc_cam2pc_world(point_cloud,extrs[i])
            #tips:点云与np矩阵转换方式
            # point_cloud=np.asarray(point_cloud.points)
            # pc=o3d.geometry.PointCloud()
            # pc.points=o3d.utility.Vector3dVector(point_cloud)
            clouds.append(point_cloud)
        o3d.visualization.draw_geometries(clouds)
    def grasp(self):
        self.grasp_pos()
if __name__ == '__main__':

    darobot=D_Robot()
    p.setRealTimeSimulation(True)
    while p.isConnected():
        desire_pos=darobot.read_gui_sliders()
        joints=darobot.calculate_ik([desire_pos[:3],desire_pos[3:6]],[desire_pos[6:9],desire_pos[6:9]])#desire_pos[6:9])

        joints[-2]=desire_pos[-2]
        joints[-1]=desire_pos[-1]
        # Use the following code to control the robot pose in real time
        # darobot.set_joint_angles(joints)
        keys = p.getKeyboardEvents()
        if ord("z") in keys and keys[ord("z")] & p.KEY_WAS_RELEASED:#take pictures
            rgbimgs,depthimgs,extrs=darobot.take_pic()
            darobot.make_3d(rgbimgs,depthimgs,extrs)
        elif ord("a") in keys and keys[ord("a")] & p.KEY_WAS_RELEASED:
            darobot.grasp_pos()

        p.stepSimulation()
        time.sleep(0.02)

