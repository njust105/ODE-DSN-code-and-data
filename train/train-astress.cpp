
#include "GRUaStress.h"
#include "CSVDataHandler.h"
#include "TimestampedSequenceDataset.h"
#include "ReduceLROnPlateau.h"
#include "AStressTrainer.h"
#include <nlohmann/json.hpp>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: train [path to configure file]" << std::endl;
        return 1;
    }
    std::string configure_path = std::string(argv[1]);

    std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;

    std::ifstream configure(configure_path);
    nlohmann::ordered_json config;
    configure >> config;
    configure.close();

    auto train_config = config["training"];

    GRUaStress model(config["model"]);

    // load data
    CSVDataHandler train_data(train_config["dataset_directory"]["training"], config["model"]["input_output"]["dim"], train_config["min_length"], true);
    CSVDataHandler val_data(train_config["dataset_directory"]["validation"], config["model"]["input_output"]["dim"], train_config["min_length"], true);

    // output scaling
    config["scaling"]["E"] = train_data.abs_max_E();
    config["scaling"]["S"] = train_data.abs_max_S();

    std::ofstream outconfigure(configure_path);
    std::ofstream log_file("train_record.txt", std::ios::app);
    if (!log_file.is_open())
    {
        std::cerr << "can't open train_record.txt" << std::endl;
        return 1;
    }

    outconfigure << config.dump(4);
    outconfigure.close();

    auto train_set = TimestampedSequenceDataset(train_data.t(), train_data.normalized_E(), train_data.normalized_S())
                         .map(torch::data::transforms::Stack<>());
    auto val_set = TimestampedSequenceDataset(val_data.t(), val_data.E() / train_data.abs_max_E(), val_data.S() / train_data.abs_max_S())
                       .map(torch::data::transforms::Stack<>());

    std::cout << "Train batch size: " << train_set.size().value() << std::endl;
    std::cout << "Validation batch size: " << val_set.size().value() << std::endl;

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
    // torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(train_config["learning_rate"]));
    // torch::optim::StepLR lr_scheduler(optimizer, train_config["decay_step"], train_config["decay_rate"]);

    ReduceLROnPlateau lr_scheduler(optimizer, 0.1, 10, 1e-2);

    // device
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // trainer
    auto trainer = make_astress_trainer(model, loss_fn, train_loader, val_loader, optimizer, lr_scheduler, device, torch::kFloat64);

    std::string dir = train_config["save_directory"];
    if (dir != "")
        dir += "/";
    // loss output
    std::ofstream loss_file(dir + "losses.csv");
    trainer.addStream(&loss_file);
    trainer.addStream(&std::cout);

    trainer.setNumEpochs(train_config["epochs"]);
    trainer.setSaveInterval(train_config["save_interval"]);

    trainer.setSaveName(dir + "model");

    trainer.checkSettings();

    int total_params = 0;
    for (const auto &param : model->parameters())
    {
        total_params += param.numel();
    }
    std::cout << "total params: " << total_params << std::endl;
    trainer.train();
    loss_file.close();

    double train_loss, val_loss;
    std::tie(train_loss, val_loss) = trainer.final_loss();
    log_file << "\n*****************************************" << std::endl;
    log_file << config.dump(4) << std::endl;
    log_file << "total params: " << total_params << std::endl;
    log_file << "train loss: " << train_loss << "\tval loss: " << val_loss << std::endl;
    log_file.close();
    return 0;
}
