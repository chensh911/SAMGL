# coding=utf-8

def load_default_param_config():

    use_pretrain_features = False

    random_projection_align = False
    input_random_projection_size = None

    merge_mode = "concat"
    target_feat_random_project_size = None
    add_self_group = False


    input_drop_rate = 0.1
    drop_rate = 0.4

    hidden_size = 512

    inner_k = 2
    squash_k = 3

    
    conv_filters = 2
    num_layers_list = [2, 0, 2]

    return squash_k, inner_k, conv_filters, num_layers_list, hidden_size, merge_mode, input_drop_rate, drop_rate, \
           use_pretrain_features, random_projection_align, input_random_projection_size, target_feat_random_project_size, add_self_group




