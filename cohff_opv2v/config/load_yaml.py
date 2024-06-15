from config.yaml_utils import load_yaml

def get_value():
    """get configuration values

    Returns:
        task_config: configuration values for tasks
    """    
    task_config = load_yaml("./config/train_config/task_config.yml") # directory of configuration yaml file
    return task_config

task_config=get_value()