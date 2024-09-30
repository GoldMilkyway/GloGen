model_configuration = {"resnet1d":
                       {"bcg":{"data_dim": 625, "last_dim": 256,}, 
                       "ppgbp": {"data_dim": 262, "last_dim": 64}, 
                       "sensors": {"data_dim": 625, "last_dim": 512}, 
                       "uci2": {"data_dim": 625, "last_dim": 256},
                       "mimic": {"data_dim": 1250, "last_dim": 512},
                       "vitaldb": {"data_dim": 1250, "last_dim": 512}},
                       "mlpbp":
                       {"bcg":{"data_dim": 625, "last_dim": 2500, }, 
                       "ppgbp": {"data_dim": 262, "last_dim": 1048}, 
                       "sensors": {"data_dim": 625, "last_dim": 2500}, 
                       "uci2": {"data_dim": 625, "last_dim": 2500}}} 