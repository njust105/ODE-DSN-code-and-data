#include "GRUaStress.h"
#include "CSVDataHandler.h"
#include <nlohmann/json.hpp>

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: test-loss [model_path] [configure_path]" << std::endl;
        return 1;
    }
    std::string model_path = std::string(argv[1]);
    std::string configure_path = std::string(argv[2]);

    std::ifstream configure(configure_path);
    nlohmann::ordered_json config;
    configure >> config;
    configure.close();

    GRUaStress model(config["model"]);
    model->to(torch::kFloat64);
    torch::load(model, model_path);

    int dim = config["model"]["input_output"]["dim"];

    CSVDataHandler data("test_data", dim, config["training"]["min_length"], true);

    double mE = config["scaling"]["E"];
    double mS = config["scaling"]["S"];

    auto E = data.E();
    auto S = data.S();
    auto t = data.t();

    torch::nn::MSELoss loss_fn(torch::kNone);

    auto timestamped_E = torch::cat({t.unsqueeze(-1), E / mE}, -1);

    torch::Tensor mask1 = (t >= 0);

    torch::Tensor output;
    std::tie(output, std::ignore) = model->forward(timestamped_E);
    torch::Tensor pred_stress = output.slice(-1, 0, dim == 3 ? 6 : 3);

    torch::Tensor loss = loss_fn->forward(pred_stress, S / mS);
    torch::Tensor mask = mask1.unsqueeze(-1).expand_as(pred_stress);

    auto test_loss = loss.masked_select(mask).mean().item<double>();

    std::cout << test_loss << std::endl;
}