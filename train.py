import configs.config as config
import utils


def compile_model(model):
    n_ds_branches = config.model.n_ds_branches
    loss = config.training.get_loss(n_ds_branches)
    metrics = config.training.get_metrics(config.data.n_outputs, n_ds_branches)
    model.compile(
        optimizer=config.training.optimizer,
        loss=loss,
        metrics=metrics,
    )


def train_model(model, train_dataloader, val_dataloader, callbacks):
    model.fit(
        train_dataloader,
        epochs=config.training.restart_epochs[-1],
        validation_data=val_dataloader,
        callbacks=callbacks,
        verbose=1,
    )


def main():
    utils.update_config_from_cli(config)
    
    train_dataloader = utils.build_dataloader(
        config.data.train_sampler_builder,
        config.data.train_sampler_args, 
        config.data.train_plugins, 
        config.data.batch_size,
        config.data.labels,
        len_factor=4
    )
    if len(train_dataloader) == 0:
        print(f"The subset is empty for {config.model.model_name}")
        return
    steps_per_epoch = len(train_dataloader)
    val_dataloader = utils.build_dataloader(
        config.data.val_sampler_builder,
        config.data.val_sampler_args, 
        config.data.val_plugins, 
        config.data.batch_size,
        config.data.labels
    )

    model_builder = config.model.model_builder
    model_args = config.model.model_args
    model = utils.build_model(
        model_builder, model_args, 
        config.data, mode="training"
    )

    # model.summary()

    compile_model(model)
    callbacks = config.training.get_callbacks(
        config.model.model_name, 
        [steps_per_epoch * epochs for epochs in config.training.restart_epochs],
        config.data.n_outputs
    )
    train_model(model, train_dataloader, val_dataloader, callbacks)


if __name__ == "__main__":
    main()
