import wandb

# useful reusable functions for Weights & Biases

def edit_hyperparameter(name, value):
    api = wandb.Api()
    # add 'change-required' tag to the run we want to change
    runs = api.runs("mgr-anomaly-tsxai/mgr-anomaly-tsxai-project", filters={'tags': 'change-required'})
    if len(runs) == 0:
        print('no runs found')
    else:
        print(f'found {len(runs)} runs')
        for run in runs:
            if name in run.config.keys():
                print('hyperparameter found')
                run.config[name] = value
                run.update()

def delete_hyperparameter(name):
    api = wandb.Api()
    # add 'change-required' tag to the run we want to change
    runs = api.runs("mgr-anomaly-tsxai/mgr-anomaly-tsxai-project", filters={'tags': 'change-required'})
    if len(runs) == 0:
        print('no runs found')
    else:
        print(f'found {len(runs)} runs')
        for run in runs:
            if name in run.config.keys():
                print('hyperparameter found')
                del run.config[name]
                run.update()

def add_hyperparameter(name, value):
    api = wandb.Api()
    # add 'change-required' tag to the run we want to change
    runs = api.runs("mgr-anomaly-tsxai/mgr-anomaly-tsxai-project", filters={'tags': 'change-required'})
    if len(runs) == 0:
        print('no runs found')
    else:
        print(f'found {len(runs)} runs')
        for run in runs:
            run.config[name] = value
            run.update()

if __name__ == '__main__':
    edit_hyperparameter("max_epochs", 500)