#include <pybind11/pybind11.h>
#include "GRUaStress.h"
#include "CSVDataHandler.h"
#include "TimestampedSequenceDataset.h"
#include "ReduceLROnPlateau.h"
#include "AStressTrainer.h"
#include <nlohmann/json.hpp>

double objective(std::string configure_path,
                 int batch_size, double learning_rate, double weight_decay,
                 int hidden_size, int input_size, int num_layers,
                 int ode_steps, int ode_num_layers, std::string integration_method,
                 bool only_return_params_num, std::string save_prefix)
{
    std::ifstream configure(configure_path);
    nlohmann::ordered_json config;
    configure >> config;
    configure.close();

    // modify config
    config["model"]["gru"]["hidden_size"] = hidden_size;
    config["model"]["gru"]["input_size"] = input_size;
    config["model"]["gru"]["num_layers"] = num_layers;

    config["model"]["ode"]["steps"] = ode_steps;
    config["model"]["ode"]["num_layers"] = ode_num_layers;
    config["model"]["ode"]["integration_method"] = integration_method;

    config["training"]["batch_size"] = batch_size;
    config["training"]["learning_rate"] = learning_rate;
    config["training"]["weight_decay"] = weight_decay;
    //

    auto train_config = config["training"];

    GRUaStress model(config["model"]);

    int total_params = 0;
    for (const auto &param : model->parameters())
    {
        total_params += param.numel();
    }

    if (only_return_params_num)
    {
        return total_params;
    }

    std::cout << "total params: " << total_params << std::endl;

    // load data
    CSVDataHandler train_data(train_config["dataset_directory"]["training"], config["model"]["input_output"]["dim"], train_config["min_length"], true);
    CSVDataHandler val_data(train_config["dataset_directory"]["validation"], config["model"]["input_output"]["dim"], train_config["min_length"], true);

    // output scaling
    config["scaling"]["E"] = train_data.abs_max_E();
    config["scaling"]["S"] = train_data.abs_max_S();

    auto train_set = TimestampedSequenceDataset(train_data.t(), train_data.normalized_E(), train_data.normalized_S())
                         .map(torch::data::transforms::Stack<>());
    auto val_set = TimestampedSequenceDataset(val_data.t(), val_data.E() / train_data.abs_max_E(), val_data.S() / train_data.abs_max_S())
                       .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set),
                                                                                            torch::data::DataLoaderOptions()
                                                                                                .batch_size(train_config["batch_size"])
                                                                                                .workers(8));
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(val_set),
                                                                                              torch::data::DataLoaderOptions()
                                                                                                  .batch_size(train_config["batch_size"])
                                                                                                  .workers(8));

    // loss function, optimizer and lr scheduler
    torch::nn::MSELoss loss_fn(torch::kNone);
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(train_config["learning_rate"]).weight_decay(train_config["weight_decay"]));

    ReduceLROnPlateau lr_scheduler(optimizer, 0.1, 5, 1e-2, 0);

    // device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // trainer
    auto trainer = make_astress_trainer(model, loss_fn, train_loader, val_loader, optimizer, lr_scheduler, device, torch::kFloat64);

    trainer.setNumEpochs(train_config["epochs"]);
    trainer.setSaveInterval(10);

    trainer.setSaveName(save_prefix + "model");

    // loss output
    std::ofstream loss_file(save_prefix + "losses.csv");
    trainer.addStream(&loss_file);

    trainer.checkSettings(false);

    trainer.train();
    loss_file.close();

    double train_loss, val_loss;
    std::tie(train_loss, val_loss) = trainer.final_loss();
    return val_loss;
}

PYBIND11_MODULE(optim_module, m)
{
    m.def("objective", &objective,
          "Objective function",
          pybind11::arg("configure_path"), pybind11::arg("batch_size"), pybind11::arg("learning_rate"),
          pybind11::arg("weight_decay"), pybind11::arg("hidden_size"), pybind11::arg("input_size"),
          pybind11::arg("num_layers"), pybind11::arg("ode_steps"), pybind11::arg("ode_num_layers"),
          pybind11::arg("integration_method"), pybind11::arg("only_return_params_num"), pybind11::arg("save_prefix"));
}
