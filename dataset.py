import torch
import utils
import numpy as np

from matplotlib import pyplot as plt
from config import Config
from torch.utils.data.dataset import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction.helper import PredictHelper
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.prediction.input_representation.agents import AgentBoxesWithFadedHistory

from nuscenes.prediction.input_representation.static_layers import StaticLayerRasterizer
from nuscenes.prediction.input_representation.interface import InputRepresentation
from nuscenes.prediction.input_representation.combinators import Rasterizer

import pickle


class NuSceneDataset(Dataset):
    def __init__(self, train_mode: bool):
        super().__init__()
        config = Config()

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.dataroot = config.dataset_path
        self.nuscenes = NuScenes(
            config.dataset_str, dataroot=self.dataroot, verbose=False
        )
        self.helper = PredictHelper(self.nuscenes)

        self.set = config.set
        self.train_mode = train_mode
        if self.set == "train":
            self.train_set = get_prediction_challenge_split(
                "train", dataroot=self.dataroot
            )
            self.val_set = get_prediction_challenge_split(
                "train_val", dataroot=self.dataroot
            )
            self.mode = "train"
        else:
            self.train_set = get_prediction_challenge_split(
                "mini_train", dataroot=self.dataroot
            )
            self.val_set = get_prediction_challenge_split(
                "mini_val", dataroot=self.dataroot
            )
            self.mode = "mini"

        self.img_layers_list = config.img_map_layers_list
        self.color_list = []
        for i in range(len(self.img_layers_list)):
            self.color_list.append((255, 255, 255))

        self.resolution = config.resolution
        self.meters_ahead = config.meters_ahead
        self.meters_behind = config.meters_behind
        self.meters_left = config.meters_left
        self.meters_right = config.meters_right

        self.num_past_hist = config.num_past_hist
        self.num_future_hist = config.num_future_hist

        self.static_layer = StaticLayerRasterizer(
            helper=self.helper,
            layer_names=self.img_layers_list,
            colors=self.color_list,
            resolution=self.resolution,
            meters_ahead=self.meters_ahead,
            meters_behind=self.meters_behind,
            meters_left=self.meters_left,
            meters_right=self.meters_right,
        )
        self.agent_layer = AgentBoxesWithFadedHistory(
            helper=self.helper,
            seconds_of_history=int(self.num_past_hist / 2),
            resolution=self.resolution,
            meters_ahead=self.meters_ahead,
            meters_behind=self.meters_behind,
            meters_left=self.meters_left,
            meters_right=self.meters_right,
        )
        self.input_repr = InputRepresentation(
            static_layer=self.static_layer,
            agent=self.agent_layer,
            combinator=Rasterizer(),
        )
        self.show_imgs = config.show_imgs
        self.save_imgs = config.save_imgs

        # for labeling
        traj_path = config.traj_set_path
        traj = pickle.load(open(traj_path, "rb"))
        self.traj = torch.Tensor(traj)

        self.num_max_agent = config.num_max_agent
        if self.save_imgs:
            if self.train_mode:
                utils.save_imgs(
                    self, self.train_set, self.set + "train", self.input_repr
                )
            else:
                utils.save_imgs(self, self.val_set, self.set + "val", self.input_repr)

    def __len__(self):
        if self.train_mode:
            return len(self.train_set)
        else:
            return len(self.val_set)

    def select_agents(self, ego_sample_token: str, ego_pose: np.ndarray):
        sample = self.helper.get_annotations_for_sample(ego_sample_token)
        mask = [
            ego_pose[0] - self.meters_left,
            ego_pose[1] + self.meters_ahead,
            ego_pose[0] + self.meters_right,
            ego_pose[1] - self.meters_behind,
        ]  # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]

        all_agents_in_sample = []
        for i in range(len(sample)):
            # for check consistency
            assert (
                ego_sample_token in sample[i]["sample_token"]
            ), "Something went wrong! Check again"

            if "vehicle" in sample[i]["category_name"]:
                all_agents_in_sample.append(sample[i])

        # 1) position filtering
        agents_in_egoframe = []
        dist_to_ego = []
        for i in range(len(all_agents_in_sample)):
            agent_pose_for_check = all_agents_in_sample[i]["translation"]
            bool_in_egoframe = (
                (agent_pose_for_check[0] > mask[0])
                and (agent_pose_for_check[0] < mask[2])
                and (agent_pose_for_check[1] < mask[1])
                and (agent_pose_for_check[1] > mask[3])
            )  # check if each agent is in the frame of ego (Type : BOOL)
            if bool_in_egoframe:
                agents_in_egoframe.append(all_agents_in_sample[i])
                dist = np.linalg.norm(
                    np.array((ego_pose[0], ego_pose[1]))
                    - np.array((agent_pose_for_check[0], agent_pose_for_check[1]))
                )
                dist_to_ego.append(dist)

        # 2) num_agents filtering (Up to num_max_agent in config.py)
        num_agents = len(agents_in_egoframe)  # before filtering
        agents = []
        if num_agents > self.num_max_agent:
            sort_idx_list = sorted(
                range(len(dist_to_ego)), key=lambda k: dist_to_ego[k]
            )
            put_list = sort_idx_list[: self.num_max_agent]
            for i in range(len(agents_in_egoframe)):
                if i in put_list:
                    agents.append(agents_in_egoframe[i])
        else:
            agents = agents_in_egoframe
        num_agents = len(agents)  # (after) filtering

        return num_agents, agents

    def get_label(self, future):
        scores = torch.full((len(self.traj),), 1e4)
        for i in range(len(self.traj)):
            if torch.norm(self.traj[i, -1] - future[-1]) < 10:
                scores[i] = torch.norm(self.traj[i] - future)
        ind = torch.argmin(scores)
        res = torch.zeros_like(scores)
        res[ind] = 1

        return res, ind

    def __getitem__(self, idx):
        if self.train_mode:
            self.dataset = self.train_set
        else:
            self.dataset = self.val_set

        #################################### Ego states ####################################
        ego_instance_token, ego_sample_token = self.dataset[idx].split("_")
        ego_annotation = self.helper.get_sample_annotation(
            ego_instance_token, ego_sample_token
        )
        ego_pose = np.array(utils.get_pose_from_annot(ego_annotation))

        ego_vel = self.helper.get_velocity_for_agent(
            ego_instance_token, ego_sample_token
        )
        ego_accel = self.helper.get_acceleration_for_agent(
            ego_instance_token, ego_sample_token
        )
        ego_yawrate = self.helper.get_heading_change_rate_for_agent(
            ego_instance_token, ego_sample_token
        )
        [ego_vel, ego_accel, ego_yawrate] = utils.data_filter(
            [ego_vel, ego_accel, ego_yawrate]
        )  # Filter unresonable data (make nan to zero)
        ego_states = np.array([ego_vel, ego_accel, ego_yawrate])

        ego_future_hist_local = self.helper.get_future_for_agent(
            instance_token=ego_instance_token,
            sample_token=ego_sample_token,
            seconds=int(self.num_future_hist / 2),
            in_agent_frame=True,
        )

        #################################### Image processing ####################################
        ego_img = self.input_repr.make_input_representation(
            instance_token=ego_instance_token, sample_token=ego_sample_token
        )

        ########################################### labeling #######################################
        label, ind = self.get_label(ego_future_hist_local)

        ego_img = torch.from_numpy(ego_img)
        ego_pose = torch.from_numpy(ego_states)
        ego_states = torch.from_numpy(ego_states)
        ego_future_hist_local = torch.from_numpy(ego_future_hist_local)

        return {
            "ego_img": ego_img.to(self.device),  # Type : np.array
            "ego_cur_pos": ego_pose.to(
                self.device
            ),  # Type : np.array([global_x,globa_y,global_yaw])                        | Shape : (3, )
            "ego_state": ego_states.to(
                self.device
            ),  # Type : np.array([vel,accel,yaw_rate]) --> local(ego's coord)          | Shape : (3, )                        | Unit : [m/s, m/s^2, rad/sec]
            "ego_future_hist_local": ego_future_hist_local.to(self.device),
            "label": (label.to(self.device), ind.to(self.device)),
        }

        # When {vel, accel, yaw_rate} is nan, it will be shown as 0
        # History List of records.  The rows decrease with time, i.e the last row occurs the farthest in the past.


if __name__ == "__main__":
    val_dataset = NuSceneDataset(train_mode=False)
    train_dataset = NuSceneDataset(train_mode=True)
    # print(dataset.__len__())
    # for i in range(dataset.__len__()):
    #     print("here is  : ", i)
    print("val_dataset:", len(val_dataset))
    print("train_dataset: ",len(train_dataset))
    d = dataset.__getitem__(10)
    print(np.shape(d["ego_img"]))
    print(np.shape(d["agent_img"]))

    
