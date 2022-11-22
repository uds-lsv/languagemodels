import os

import wandb


class WandbLogger():
    def __init__(self, project_name, output_dir, run_name, group_name, config):
        if os.environ["WANDB_DISABLED"] == "false":
            wandb.init(
                project=project_name,
                dir=output_dir,
                name=run_name,
                group=group_name,
                config=config)

            # define x-axis
            wandb.define_metric("global_step")
            wandb.define_metric(
                "*", step_metric="global_step", step_sync=True)

    def log(self, logs, step):
        wandb.log(data=logs, step=step)
