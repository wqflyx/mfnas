class DefaultConfig(object):
    # system related
    use_gpu = True
    gpu = 1
    seed = None  # None or number

    # dataset related
    base2_model_dir = "./model/"
    base2_genotype_dir = "./results/"
    dataset_root = "/public/data/"
    batch_size = 96
    train_portion = 0.6  # 0.6
    worker_nums = 2  # how many workers for loading data
    cutout = True
    cutout_length = 16

    # result related
    report_freq = 200

    init_channels = 16  # 50, 36 for large dataset; 16 for small datasets
    layers = 8  # 14, 20 for large dataset; 8 for small dataset
    module_nodes = 4
    input_links_per_node = 2
    primitive_nodes = 7  # conv3, conv5, avg, max, x, dil3, dil5
    drop_path_prob = 0
    pool_position = [layers // 3, 2 * layers // 3]
    auxiliary = False
    auxiliary_weight = 0.4
    sss_ps = 150  # search space switch picture size
    flow_fix_input = True

    # child related
    learning_rate = 0.025
    learning_rate_min = 0.0001
    momentum = 0.9
    weight_decay = 3e-9
    grad_clip = 5

    max_search_epoch = 600
    tasks = [(1, 1)]

    primitive = [
        'none',
        'max_pool_3x3',
        'avg_pool_3x3',
        'skip_connect',
        'sep_conv_3x3',
        'sep_conv_5x5',
        'dil_conv_3x3',
        'dil_conv_5x5'
    ]

    enas2darts = {
        3: 'max_pool_3x3',
        2: 'avg_pool_3x3',
        4: 'skip_connect',
        0: 'sep_conv_3x3',
        1: 'sep_conv_5x5',
        5: 'dil_conv_3x3',
        6: 'dil_conv_5x5'
    }

    def parse(self, kwargs):
        '''
        :param self:
        :param kwargs: dict to update config
        :return:
        '''

        for k, v in kwargs.items():
            if not hasattr(self, k):
                print("warning: opt has not attribute %s" % k)
            setattr(self, k, v)
        print("user config:")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith("__"):
                print(k, getattr(self, k))
