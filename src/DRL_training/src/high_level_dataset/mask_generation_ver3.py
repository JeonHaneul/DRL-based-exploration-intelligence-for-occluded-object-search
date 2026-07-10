## Usage
## ./isaaclab.sh -p source/standalone/shelf_env/mask_generation_ver3.py --target_object can_2 --enable_camera --save --num_img 100

import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--target_object",
    type=str,
    default="cup_1",
    help="Name of the target object",
)
parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        "The viewport will always initialize with the perspective of camera 0."
    ),
)
parser.add_argument(
    "--num_img",
    type=int,
    default="10",
    help="Number of images to generate",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import numpy as np
import os
import random

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
import numpy as np

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth
from omni.isaac.lab.utils import convert_dict_to_backend

from scipy.spatial.transform import Rotation as R

import cv2

target_row_index = 4 # ???�? ?���? ?���? (5x5 배열?�� 경우 1~5) / (4x3 배열?�� 경우 1~3)
spawn_probability = 0.15  # Probability of placing the target object in column 1 or 2
visibility_probability = 0.2  # Probability of ensuring visibility of the target object


class ENV_Cfg:
    def __init__(self):
        self.target_row_index = target_row_index
        
        ## 4x5 ?��반환�? / y�? 간격 15�? �?�? / ?���? 갈수�? 2?�� 증�??(?��?�� set?��?��?��) / speedrack ?��?�� - 최종
        self.shelf = [
            [[0.15, 0.30, 1.05], [0.15, 0.15, 1.05], [0.15, 0.0, 1.05], [0.15, -0.15, 1.05], [0.15, -0.30, 1.05]],
            [[0.05, 0.315, 1.05], [0.05, 0.16, 1.05], [0.05, 0.0, 1.05], [0.05, -0.16, 1.05], [0.05, -0.315, 1.05]],
            [[-0.05, 0.33, 1.05], [-0.05, 0.17, 1.05], [-0.05, 0.0, 1.05], [-0.05, -0.17, 1.05], [-0.05, -0.33, 1.05]],
            [[-0.15, 0.345, 1.05], [-0.15, 0.18, 1.05], [-0.15, 0.0, 1.05], [-0.15, -0.18, 1.05], [-0.15, -0.345, 1.05]]
        ]
        
        self.items = ["can_1", "can_2", "can_3", "can_4", "cup_1", "cup_2", "cup_3", "cup_4", "mug_1", "mug_2", "mug_3", "mug_4", "bottle_1", "bottle_2", "bottle_3", "bottle_4"]  # ?��?��?�� ?���? 배열
        # 카테고리�? ?��?��?�� 분류
        self.category_mapping = {
            "cup": ["cup_1", "cup_2", "cup_3", "cup_4"],
            "mug": ["mug_1", "mug_2", "mug_3", "mug_4"],
            "bottle": ["bottle_1", "bottle_2", "bottle_3", "bottle_4"],
            "can": ["can_1", "can_2", "can_3", "can_4"],
        }
        self.placement_list = []
        
    def design_scene(self):
        """Designs the scene by spawning ground plane, light, objects and meshes from usd files"""
        # Ground-plane
        cfg_ground = sim_utils.GroundPlaneCfg()
        cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
        
        # spawn distant light
        cfg_light_dome = sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(1.0, 1.0, 1.0),
        )
        cfg_light_dome.func("/World/lightDistant", cfg_light_dome, translation=(-5, 0, 10))
        
        cfg_light_dome2 = sim_utils.DomeLightCfg(
            intensity=2700.0,
            color=(1.0, 1.0, 1.0),
        )
        cfg_light_dome2.func("/World/lightDistant2", cfg_light_dome2, translation=(1.2, 0.0, 1.4))

        # spawn a usd file of a shelf into the scene
        rack_cfg = RigidObjectCfg(
            prim_path="/World/Rack",
            spawn=sim_utils.UsdFileCfg(usd_path=f"omniverse://localhost/Library/Shelf/Arena/speedrack.usd", mass_props=MassPropertiesCfg(mass=500)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            debug_vis=False,
        )
        rack = RigidObject(cfg=rack_cfg)

        # Create prims for each position in the shelf
        for shelf_idx, shelf_row in enumerate(self.shelf):
            for col_idx, position in enumerate(shelf_row):
                prim_utils.create_prim(f"/World/shelf_{shelf_idx}_col_{col_idx}", "Xform", translation=position)

        # Spawn objects in the scene
        scene_entities = self.obj_spawn()
        
        # Define camera sensor and add it to the scene
        camera = self.define_sensor()
        scene_entities["camera"] = camera
        
        return scene_entities

    def get_category(self, item_name):
        for category, items in self.category_mapping.items():
            if item_name in items:
                return category
        return None

    def obj_spawn(self) -> dict:
        scene_entities = {}
        self.placement_list = []

        # ?��?��?�� ?��?�� 기�???�� target_row_index�? 배열 ?��?��?���? �??��
        if np.random.rand() < spawn_probability:
            adjusted_target_row_index = np.random.choice([0, 1, 2])  # �? 번째(0) ?��?�� ?�� 번째(1), ?��번째(2)?�� ?��?��
        else:
            adjusted_target_row_index = self.target_row_index - 1  # ?��?��?�� 1~5�? ?��?��?�� 값을 0~4�? �??��

        # ???�? ?���? �? 객체 ?��?��
        self.target_position = (adjusted_target_row_index, np.random.randint(0, len(self.shelf[adjusted_target_row_index])))  # (row_idx, col_idx)
        target_object = args_cli.target_object  # argparse?��?�� 받�?? ???�? 객체 ?���?

        # ???�? 객체?�� ????�� USD ?��?�� 경로 ?��?��
        usd_path_mapping = {
            "cup_1": "omniverse://localhost/Library/Shelf/Object/Cup_1_9.usd",
            "cup_2": "omniverse://localhost/Library/Shelf/Object/Cup_2_9.usd",
            "cup_3": "omniverse://localhost/Library/Shelf/Object/Cup_4_9.usd",
            "cup_4": "omniverse://localhost/Library/Shelf/Object/Cup_5_9.usd",
            "mug_1": "omniverse://localhost/Library/Shelf/Object/Mug_2_9.usd",
            "mug_2": "omniverse://localhost/Library/Shelf/Object/Mug_3_9.usd",
            "mug_3": "omniverse://localhost/Library/Shelf/Object/Mug_4_9.usd",
            "mug_4": "omniverse://localhost/Library/Shelf/Object/Mug_5_9.usd",
            "bottle_1": "omniverse://localhost/Library/Shelf/Object/Bottle_6.usd",
            "bottle_2": "omniverse://localhost/Library/Shelf/Object/Bottle_7.usd",
            "bottle_3": "omniverse://localhost/Library/Shelf/Object/Bottle_8.usd",
            "bottle_4": "omniverse://localhost/Library/Shelf/Object/Bottle_9.usd",
            "can_1": "omniverse://localhost/Library/Shelf/Object/Can_6.usd",
            "can_2": "omniverse://localhost/Library/Shelf/Object/Can_10.usd",
            "can_3": "omniverse://localhost/Library/Shelf/Object/Can_8.usd",
            "can_4": "omniverse://localhost/Library/Shelf/Object/Can_9.usd",
        }

        # ???�? ?��치의 Prim 경로 ?��?��
        target_row_idx, target_col_idx = self.target_position
        target_prim_path = f"/World/shelf_{target_row_idx}_col_{target_col_idx}"

        if not prim_utils.is_prim_path_valid(target_prim_path):
            prim_utils.create_prim(target_prim_path, "Xform")  # Prim ?��?���? ?��?�� (좌표�? ?��?��)

        # ???�? ?��브젝?�� 배치 (?��?���? 추�??)
        target_noise = np.random.uniform(-0.01, 0.01, size=2)  # x??? y 축에 각각 최�?? 1cm ?��?���? 추�??
        target_rotation_z = np.random.uniform(0, 360)  # Z�? ?��?�� (0~360?��)
        target_position_offset = [
            target_noise[0],  # ?��?���? 값만 추�??
            target_noise[1],
            0.0,  # z 좌표�? 0?���? 고정
        ]
        target_rotation_quaternion = R.from_euler('z', target_rotation_z, degrees=True).as_quat()  # 쿼터?��?�� 계산
        target_rotation_quaternion = [target_rotation_quaternion[3],  # w
                                  target_rotation_quaternion[0],  # x
                                  target_rotation_quaternion[1],  # y
                                  target_rotation_quaternion[2]]  # z

        obj = RigidObject(cfg=RigidObjectCfg(
            prim_path=f"{target_prim_path}/{target_object}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path_mapping[target_object],
                scale=(1.0, 1.0, 1.0),
                semantic_tags=[("class", target_object)],
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=target_position_offset, rot=target_rotation_quaternion),
        ))
        scene_entities[f"shelf_{target_row_idx}_col_{target_col_idx}"] = obj

        # 카테고리 구분
        target_category = self.get_category(target_object)
        same_category_items = self.category_mapping[target_category].copy()
        same_category_items.remove(target_object)
        random.shuffle(same_category_items)

        # ?��?��?�� 카테고리 (0.5): ?���? 카테고리 중에?�� ?��?��?�� 카테고리 ?��?��
        similar_category = None
        if target_category in ["cup", "mug"]:
            similar_category = "mug" if target_category == "cup" else "cup"
        elif target_category in ["bottle", "can"]:
            similar_category = "can" if target_category == "bottle" else "bottle"
        similar_category_items = self.category_mapping[similar_category].copy()
        random.shuffle(similar_category_items)

        # ?���? 카테고리 (0.1): ?��머�?? 카테고리
        other_categories = set(self.category_mapping.keys()) - {target_category, similar_category}
        other_category_items = []
        for cat in other_categories:
            other_category_items.extend(self.category_mapping[cat])
        random.shuffle(other_category_items)

        # ?���? ?���? 리스?�� ?��?��
        all_positions = [(row_idx, col_idx) for row_idx in range(len(self.shelf)) for col_idx in range(len(self.shelf[0]))]
        all_positions.remove((target_row_idx, target_col_idx))  # ???�? ?���? ?��?��
        
        # ?��?�� ?��률로 ???�? 객체 ?��쪽을 비우?���? ?��
        empty_positions = set()
        if np.random.rand() < visibility_probability:
            for row_idx in range(target_row_idx - 1, -1, -1):  # ???�? 객체보다 ?���? ?��(row)
                empty_positions.add((row_idx, target_col_idx))

        # ?��치별�? 배치?�� ?��브젝?�� 리스?�� ?��?��
        placement_list = []
        placement_list.append(((target_row_idx, target_col_idx), target_object))
        
        for pos in empty_positions:
            if pos in all_positions:
                all_positions.remove(pos)
        
        # ?���? ?��?��?�� ?��치�?? 추적?���? ?��?�� 집합
        used_positions = {self.target_position}  # ???�? ?���? 추�??
        
        used_positions.update(empty_positions)

        def place_items_with_weights(items, candidate_positions, position_weights):
            """?��?��?��?�� �?중치 기반?���? 배치?���? 중복 발생 ?�� ?���? ?��?��?�� ?��리�?? ?��?��?��."""
            while items and candidate_positions:
                # �?중치 기반?���? ?���? ?��?��
                weighted_pos = random.choices(
                    population=candidate_positions,
                    weights=position_weights,
                    k=1
                )[0]

                if weighted_pos not in used_positions:
                    # 중복?���? ?��??? 경우 배치
                    item = items.pop(0)
                    placement_list.append((weighted_pos, item))
                    used_positions.add(weighted_pos)
                    
                    # ?��?��?�� ?��치�?? ?��보�?? �?중치?��?�� ?���?
                    idx = candidate_positions.index(weighted_pos)
                    candidate_positions.pop(idx)
                    position_weights.pop(idx)
                else:
                    # 중복?�� 경우 ?��보�?? �?중치?��?�� ?��?�� ?��치만 ?���?
                    idx = candidate_positions.index(weighted_pos)
                    candidate_positions.pop(idx)
                    position_weights.pop(idx)


        # 같�?? 카테고리 (0.8) 배치
        same_category_positions = []
        for row_idx in range(adjusted_target_row_index - 1, -1, -1):  # ???겟보?�� ?���?(?�� 번호�? ?��??? 방향)
            for col_offset in [-1, 0, 1]:  # ???�? ?�� 주�???�� �?(-1), ?���?(0), ?��(1)
                col_idx = target_col_idx + col_offset  # ?�� 계산
                if 0 <= col_idx < len(self.shelf[0]):  # ?��?��?�� ?��?���? ?��?��
                    same_category_positions.append((row_idx, col_idx))  # ?���? ????��

        # 중심 ?��?�� ?�� ?��??? �?중치�? �??��
        position_weights = [3.0 if pos[1] == target_col_idx else 1.0 for pos in same_category_positions]
        place_items_with_weights(same_category_items, same_category_positions, position_weights)


        # ?��?��?�� 카테고리 (0.5) 배치
        similar_category_positions = []
        similar_cols = [target_col_idx - 1, target_col_idx + 1]
        position_weights = []

        for col_idx in similar_cols:
            if 0 <= col_idx < len(self.shelf[0]):
                for row_idx in range(len(self.shelf)):
                    similar_category_positions.append((row_idx, col_idx))
                    position_weights.append(3.0)

                    # �?, ?���? ?��?��
                    adj_col_idx = col_idx + (1 if col_idx == target_col_idx - 1 else -1)
                    if 0 <= adj_col_idx < len(self.shelf[0]):
                        similar_category_positions.append((row_idx, adj_col_idx))
                        position_weights.append(1.0)

        place_items_with_weights(similar_category_items, similar_category_positions, position_weights)

        # 카테고리 0.8�? 0.5?��?�� ?��?��?�� ?�� 추적
        used_columns = {target_col_idx}  # ???�? ?�� ?��?��
        used_columns.update([pos[1] for pos in same_category_positions])  # 0.8?��?�� ?��?��?�� ?�� 추�??
        used_columns.update([pos[1] for pos in similar_category_positions])  # 0.5?��?�� ?��?��?�� ?�� 추�??

        # ?���? 카테고리 (0.1) 배치
        other_category_positions = []
        available_columns = [col_idx for col_idx in range(len(self.shelf[0])) if col_idx not in used_columns]

        for col_idx in available_columns:  # ?��?��?���? ?��??? ?��?��?���? ?��?��
            for row_idx in range(len(self.shelf)):
                other_category_positions.append((row_idx, col_idx))

        position_weights = [1.0] * len(other_category_positions)  # 균등 �?중치
        place_items_with_weights(other_category_items, other_category_positions, position_weights)


        # ?��?�� ?��브젝?�� 배치
        for (row_idx, col_idx), object_name in placement_list:
            self.placement_list.append(((row_idx, col_idx), object_name))
            prim_path = f"/World/shelf_{row_idx}_col_{col_idx}"

            if not prim_utils.is_prim_path_valid(prim_path):
                prim_utils.create_prim(prim_path, "Xform")  # ?���? Prim ?��?��

            # ?��브젝?�� 배치 (?��?���? 추�??)
            noise = np.random.uniform(-0.01, 0.01, size=2)  # x??? y 축에 각각 최�?? 1cm ?��?���? 추�??
            rotation_z = np.random.uniform(0, 360)  # Z�? ?��?�� (0~360?��)
            position_offset = [
                noise[0],  # ?��?���? 값만 추�??
                noise[1],  
                0.0,  # z 좌표�? 0?���? 고정
            ]
            rotation_quaternion = R.from_euler('z', rotation_z, degrees=True).as_quat()
            rotation_quaternion = [rotation_quaternion[3],  # w
                                rotation_quaternion[0],  # x
                                rotation_quaternion[1],  # y
                                rotation_quaternion[2]]  # z

            usd_path = usd_path_mapping[object_name]

            # ?��브젝?�� ?��?��
            obj = RigidObject(cfg=RigidObjectCfg(
                prim_path=f"{prim_path}/{object_name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    scale=(1.0, 1.0, 1.0),
                    semantic_tags=[("class", object_name)],
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=position_offset, rot=rotation_quaternion),
            ))
            scene_entities[f"shelf_{row_idx}_col_{col_idx}"] = obj

        # 배치 ?���? ?�� 최종 ?��버깅
        print("[DEBUG] Final placement_list contents:")
        for pos, obj in placement_list:
            print(f"  Position: {pos}, Object: {obj}")

        print("[DEBUG] Final used_positions contents:")
        print(used_positions)

    
        return scene_entities

    def reset_scene(self, entities: dict):
        """Reset the scene configuration"""

        for key in list(entities.keys()):
            if key != "camera":
                prim_utils.delete_prim(entities[key].cfg.prim_path)
                del entities[key]

        new_entities = self.obj_spawn()
        entities.update(new_entities)

        return entities

    def define_sensor(self,) -> Camera:
        """Defines the camera sensor to add to the scene."""
        # Setup camera sensor
        # In contrast to the ray-cast camera, we spawn the prim at these locations.
        # This means the camera sensor will be attached to these prims.
        prim_utils.create_prim("/World/Origin_00", "Xform")
        camera_cfg = CameraCfg(
            prim_path="/World/Origin_.*/CameraSensor",
            update_period=0,
            height=480,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "distance_to_camera",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ],
            colorize_semantic_segmentation=True,
            colorize_instance_id_segmentation=True,
            colorize_instance_segmentation=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
        )
        # Create camera
        camera = Camera(cfg=camera_cfg)

        return camera


def generate_mask_with_shelf_info(camera, entities, target_object, env_cfg, output_dir, frame_count):
    """
    Corrected algorithm to find the front-most objects in each column using placement_list.
    """
    try:
        # Step 1: Extract semantic_segmentation data
        placement_list = env_cfg.placement_list
        camera_info = camera.data.info[0].get("semantic_segmentation", {})
        # print("[DEBUG] Semantic segmentation info:", camera_info)

        if not camera_info:
            print("[ERROR] 'semantic_segmentation' data is not available.")
            return

        id_to_labels = camera_info["idToLabels"]
        # print(f"[DEBUG] semantic_segmentation idToLabels: {id_to_labels}")

        # Step 2: Filter objects and deduplicate
        filtered_objects = {}
        for rgba_str, data in id_to_labels.items():
            object_class = data.get("class", "UNKNOWN")
            rgba = tuple(map(int, rgba_str.strip("()").split(",")))
            if object_class not in ["BACKGROUND", "UNLABELLED"]:
                filtered_objects[rgba] = object_class

        print(f"[DEBUG] Filtered objects: {filtered_objects}")

        # Step 3: Prepare instance segmentation data
        instance_data = camera.data.output["semantic_segmentation"]
        if isinstance(instance_data, torch.Tensor):
            instance_data = instance_data.cpu().numpy()
        if instance_data.ndim == 4 and instance_data.shape[0] == 1:
            instance_data = instance_data[0]

        mask_height, mask_width = instance_data.shape[:2]
        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)

        print("[DEBUG] Using placement_list for mask generation:")
        for pos, obj in placement_list:
            print(f"    Position: {pos}, Object: {obj}")

        # Step 4: Identify front-most objects per column using placement_list
        front_objects = {}
        for (row, col), object_name in placement_list:
            # Update front-most object if the column is not in the dictionary
            # or the current object has a smaller row value
            if col not in front_objects or row < front_objects[col]["row"]:
                front_objects[col] = {"row": row, "name": object_name}

        print("[DEBUG] Updated Front objects per column with row comparison:")
        for col, obj_info in sorted(front_objects.items()):
            print(f"    Column: {col}, Row: {obj_info['row']}, Object: {obj_info['name']}")

        # Step 5: Define categories and similarities
        target_category = env_cfg.get_category(target_object)
        similar_category = None
        if target_category in ["cup", "mug"]:
            similar_category = "mug" if target_category == "cup" else "cup"
        elif target_category in ["bottle", "can"]:
            similar_category = "can" if target_category == "bottle" else "bottle"

        print(f"[DEBUG] Target Category: {target_category}, Similar Category: {similar_category}")

        # Step 6: Assign mask values based on category similarity
        for col, obj_info in front_objects.items():
            object_name = obj_info["name"]
            obj_category = env_cfg.get_category(object_name)

            # Find the RGBA value for this object
            rgba_value = None
            for rgba, class_name in filtered_objects.items():
                if class_name == object_name:
                    rgba_value = np.array(rgba[:3], dtype=instance_data.dtype)
                    break

            if rgba_value is None:
                print(f"[WARNING] No RGBA value found for object: {object_name}")
                continue

            # Determine mask value based on category similarity
            if object_name == target_object:
                mask_value = 255  # Target object
            elif obj_category == target_category:
                mask_value = int(255 * 0.8)  # Same category
            elif obj_category == similar_category:
                mask_value = int(255 * 0.5)  # Similar category
            else:
                mask_value = int(255 * 0.2)  # Other category

            print(f"[DEBUG] Assigning mask value {mask_value} to object '{object_name}' with RGBA: {rgba_value}")

            # Apply the mask only to the pixels matching the object's RGBA
            matches = np.all(instance_data[..., :3] == rgba_value, axis=-1)
            mask[matches] = mask_value

        # Step 7: Save mask
        num_img = int(frame_count)
        output_dir = os.path.join(output_dir, "mask/")
        mask_image_path = os.path.join(output_dir, f"mask_{target_object}_frame_{num_img}.png")
        os.makedirs(os.path.dirname(mask_image_path), exist_ok=True)
        from PIL import Image
        Image.fromarray(mask).save(mask_image_path)
        print(f"[INFO] Mask saved to {mask_image_path}")

    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")





def run_simulator(sim: sim_utils.SimulationContext, entities: dict, cfg: ENV_Cfg):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    count_2 = 0

    # Extract entities for simplified notation
    camera: Camera = entities["camera"]

    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera", f"{args_cli.target_object}","scene")
    os.makedirs(output_dir, exist_ok=True)
    
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
    )

    # Camera positions, targets, orientations
    camera_positions = torch.tensor([[1.18, 0.0, 1.27]], device=sim.device) # [1.05, 0.0, 0.85] - 고릴�? ?�� / [1.18, 0.0, 1.27] - ?��?��?�� ?�� / [1.175, 0.0, 1.29] - ?��?��?�� ?��2
    camera_targets = torch.tensor([[0.0, 0.0, 1.31]], device=sim.device) # [0.0, 0.0, 0.85] - 고릴�? ?�� / [0.0, 0.0, 1.31] - ?��?��?�� ?�� / [0.0, 0.0, 1.29] - ?��?��?�� ?��2

    camera.set_world_poses_from_view(camera_positions, camera_targets)
    
    # Index of the camera to use for visualization and saving
    camera_index = args_cli.camera_id

    # print("[INFO] Checking camera data availability...")
    # try:
    #     camera_data = camera.data.output
    #     print("[DEBUG] Available camera data keys:", camera_data.keys())
    # except Exception as e:
    #     print(f"[ERROR] Failed to access camera data during initialization: {e}")
    #     return

    # Simulate physics
    while simulation_app.is_running():
        sim.step()
        camera.update(dt=sim.get_physics_dt())

        sim_time += sim_dt
        count += 1
        num_img = int(args_cli.num_img)
        if count_2 >= num_img:
            raise RuntimeError

        if count % 100 == 0:
            print(f"[INFO] Processing frame at count {count}...")
            count_2 += 1
            # print("camera info : ",camera.data.info[0]["instance_id_segmentation_fast"])
            # print("camera output : ",camera.data.output["instance_segmentation_fast"])
            
            
            # Extract camera data
            if args_cli.save:
                # Save images from camera at camera_index
                # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
                # tensordict allows easy indexing of tensors in the dictionary
                single_cam_data = convert_dict_to_backend(
                {k: v[camera_index] for k, v in camera.data.output.items()}, backend="numpy"
            )
                
                generate_mask_with_shelf_info(
                camera=camera,
                entities=entities,
                target_object=args_cli.target_object,
                env_cfg=cfg,  # ENV_Cfg ?��?��?��?���? ?��?��
                output_dir=output_dir,
                frame_count=count_2,
            )

                # Extract the other information
                single_cam_info = camera.data.info[camera_index]

                # Pack data back into replicator format to save them using its writer
                rep_output = {"annotators": {}}
                for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                # Save images
                # Note: We need to provide On-time data for Replicator to save the images.
                rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
                rep_writer.write(rep_output)

            entities = cfg.reset_scene(entities)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    env = ENV_Cfg()
    sim.set_camera_view(eye=[1.5, 0.0, 1.5], target=[0.0, 0.0, 1.3])

    # Design scene
    scene_entities = env.design_scene()
    sim.reset()
    print("[INFO]: Setup complete...")

    # Run the simulator and generate masks
    run_simulator(sim, scene_entities, env)


if __name__ == "__main__":
    main()
    simulation_app.close()
