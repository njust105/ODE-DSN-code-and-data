import io
import optuna
import contextlib
import sys
# fmt: off
sys.path.append('../train/build/release')
import optim_module
# fmt: on


def objective(trial, config_path):
    batch_size = trial.suggest_int("batch_size", 16, 256)
    learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    integration_method = trial.suggest_categorical(
        "integration_method", ["RK2", "RK4"])
    hidden_size = trial.suggest_int("hidden_size", 8, 128)
    input_size = trial.suggest_int("input_size", 6, 32)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    ode_steps = trial.suggest_int("ode_steps", 2, 10)
    ode_num_layers = trial.suggest_int("ode_num_layers", 3, 15)

    total_params = optim_module.objective(
        config_path,
        batch_size,
        learning_rate,
        weight_decay,
        hidden_size,
        input_size,
        num_layers,
        ode_steps,
        ode_num_layers,
        integration_method,
        True,
        "optim_astress/"+str(trial.number)+"-"
    )

    if total_params > 1e5:
        return 1e-9*total_params

    output_buffer = io.StringIO()

    with contextlib.redirect_stdout(output_buffer):
        val_loss = optim_module.objective(
            config_path,
            batch_size,
            learning_rate,
            weight_decay,
            hidden_size,
            input_size,
            num_layers,
            ode_steps,
            ode_num_layers,
            integration_method,
            False,
            "optim_astress/"+str(trial.number)+"-"
        )

    cpp_output = output_buffer.getvalue()
    print("val_loss", val_loss, "total_params", total_params)

    return val_loss + 1e-10*total_params


if __name__ == "__main__":

    study = optuna.create_study(
        study_name="optim_astress",
        storage="sqlite:///optim_parameters_astress.db",
        load_if_exists=True,
        direction="minimize"
    )

    config_path = "configure.json"
    n_trials = 10

    study.optimize(lambda trial: objective(trial, config_path),
                   n_trials=n_trials, show_progress_bar=True)

    print("Best trials:", study.best_trial.number)
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)
