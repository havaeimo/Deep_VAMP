import whetlab
import generate_yaml
import argparse
access_token= '31ca9b27-81c2-42e9-a141-45047727050f'


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Whetlab optimization of hyperparameters')
    parser.add_argument('gpu', type=str,
                        help='example:gpu0')
    args = parser.parse_args()




    scientist = whetlab.Experiment(name="deep_vamp",
                               access_token=access_token)
    gpu = args.gpu
    for i in range(20):
        param = scientist.suggest()
	performance = generate_yaml.fit_model(param,gpu)
	scientist.update(param, performance)
