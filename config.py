class Config(object):
    def __init__(self):
        self.set = "mini"  # 'train' or 'mini'
        self.dataset_str = "v1.0-mini"
        self.dataset_path = "/home/mmc-server3/Server/dataset/NuScenes"
        self.device = "cuda:3"

        # Agent history
        self.num_past_hist = 10
        self.num_future_hist = 12

        # Image Processing
        self.show_imgs = True
        self.save_imgs = False
        self.img_map_layers_list = [
            "drivable_area",
            "road_segment",
            "road_block",
            "lane",
            "ped_crossing",
            "road_divider",
            "lane_divider",
            "traffic_light",
        ]
        self.resolution = 0.1  # [meters/pixel]
        self.meters_ahead = 40
        self.meters_behind = 10
        self.meters_left = 25
        self.meters_right = 25

        # Agent Processing
        self.num_max_agent = 4

        # For Training
        self.batch_size = 1

        # For labeling
        self.traj_set_path = (
            "./nuscenes-prediction-challenge-trajectory-sets/epsilon_2.pkl"
        )

        self.alpha = 1
        self.beta = 2
        self.epoch = 50

        self.lr = 0.0005
        self.beta1 = 0.5
