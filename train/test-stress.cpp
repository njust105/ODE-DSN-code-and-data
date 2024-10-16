#include "GRUStress.h"
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

    GRUStress model(config["model"]);
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

    int inse = 0;

    auto zz = torch::zeros({1, inse, E.size(-1)+1}, E.options());

    timestamped_E = torch::cat({zz, timestamped_E}, -2);


    auto pred_S = std::get<0>(model->forward(timestamped_E));

    pred_S = pred_S.slice(1, inse);

    pred_S *= mS;

    std::ofstream file("compare_cout.csv");

    std::vector<std::string> S_headers = (dim == 2) ? std::vector<std::string>{"S11", "S22", "S12"}
                                                    : std::vector<std::string>{"S11", "S22", "S33", "S23", "S13", "S12"};

    file << "Time";
    for (auto &s : S_headers)
    {
        file << "," << s << "," << "P" + s;
    }
    file << "\n";
    
    for (size_t i = 0; i < S_headers.size()+1; i++)
    {
       file << "0" << ",";
    }
    file << "\n";

    for (size_t i = 0; i < t.size(1); i++)
    {
        file << t[0][i].item<double>();

        for (size_t j = 0; j < S_headers.size(); j++)
        {
            file << "," << S[0][i][j].item<double>() << "," << pred_S[0][i][j].item<double>();
        }

        file << "\n";
    }
    file.close();
}