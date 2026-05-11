from usecase import Usecase
from usecase_config import UsecaseConfig

def test_usecase_config():
    for uc_cfg in UsecaseConfig.from_dataset_csv("dataset.csv"):
        print(uc_cfg.to_json())

    uc_cfg = UsecaseConfig.from_file("uc_config.ini")
    uc_cfg.to_file("uc_config_out.ini")

def test_usecase():
    uc = Usecase()
    uc.graph
    
if __name__ == "__main__":
    test_usecase()