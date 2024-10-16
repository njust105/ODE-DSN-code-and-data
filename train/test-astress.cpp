#include "GRUaStress.h"
#include "CSVDataHandler.h"
#include <rapidcsv.h>
#include <nlohmann/json.hpp>
#include "Tools.h"

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: test [model file] [configure file] [target csv]" << std::endl;
        return 1;
    }
    std::string model_path = std::string(argv[1]);
    std::string configure_path = std::string(argv[2]);
    std::string csv_path = std::string(argv[3]);

    std::ifstream configure(configure_path);
    nlohmann::ordered_json config;
    configure >> config;
    configure.close();

    GRUaStress model(config["model"]);
    model->to(torch::kFloat64);
    torch::load(model, model_path);

    int dim = config["model"]["input_output"]["dim"];

    CSVDataHandler data(dim);
    data.readCSVFile(csv_path);

    double mE = config["scaling"]["E"];
    double mS = config["scaling"]["S"];

    auto E = data.E();
    auto S = data.S();
    auto t = data.t();

    auto timestamped_E = torch::cat({t.unsqueeze(-1), E / mE}, -1);

    auto pred_output = std::get<0>(model->forward(timestamped_E));

    auto pred_S = pred_output.slice(-1, 0, dim == 2 ? 3 : 6) * mS;

    std::ofstream file("compare_cout.csv");

    std::vector<std::string> S_headers = (dim == 2) ? std::vector<std::string>{"S11", "S22", "S12"}
                                                    : std::vector<std::string>{"S11", "S22", "S33", "S23", "S13", "S12"};

    std::vector<std::string> E_headers = (dim == 2) ? std::vector<std::string>{"E11", "E22", "E12"}
                                                    : std::vector<std::string>{"E11", "E22", "E33", "E23", "E13", "E12"};

    file << "Time";

    for (unsigned int i = 0; i < S_headers.size(); i++)
    {
        file << "," << S_headers[i];
    }
    for (unsigned int i = 0; i < S_headers.size(); i++)
    {
        file << "," << "P" + S_headers[i];
    }
    file << "\n";

    file << "0";
    for (size_t i = 0; i < 2 * S_headers.size() + 1; i++)
    {
        file << "," << "0";
    }
    file << "\n";

    for (size_t i = 0; i < t.size(1); i++)
    {
        file << t[0][i].item<double>();

        for (size_t j = 0; j < S_headers.size(); j++)
        {
            file << "," << S[0][i][j].item<double>();
        }
        for (size_t j = 0; j < S_headers.size(); j++)
        {
            file << "," << pred_S[0][i][j].item<double>();
        }
        file << "\n";
    }
    file.close();
}