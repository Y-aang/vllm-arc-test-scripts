import pickle


class Query_wild(object):
    def __init__(self, qps=1.0, data_path=None, conv_or_api=None, scale=1):
        self.qps = qps
        self.query_time = 0
        self.max_prompt_len = 2048  # TODO 1024
        self.max_gen_len = 1     # TODO: decode length 1024 128
        # self.zipf_param = 1.1
        # self.gamma_shape = 0.25
        # self.gamma_scale = 2
        self.query_id = 0
        # self.gamma_shape_dict = dict()
        self.data_path = data_path
        # self.conv_or_api = conv_or_api
        self.scale = scale
        self.processed_data = read_pickle(self.data_path)


    def get_query(self):
        prompt = self.processed_data[self.query_id]['fake_tokens']
        prompt_len = self.processed_data[self.query_id]['input_length']
        sampled_prompt_len = self.processed_data[self.query_id]['output_length']
        delta_time = self.processed_data[self.query_id]['timestamp'] - self.query_time
        
        self.query_time = self.processed_data[self.query_id]['timestamp']
        self.query_id += 1

        return [prompt, prompt_len, self.max_gen_len, sampled_prompt_len, self.max_gen_len, delta_time, self.query_time]

def read_pickle(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data