#pragma once
#include <torch/torch.h>
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#elif __has_include("nlohmann/json.h")
#include "nlohmann/json.h"
#else
#error "nlohmann/json header file not found"
#endif

struct ODESolverImpl : torch::nn::Module
{
    ODESolverImpl(nlohmann::json options) : options(options)
    {
        steps = options["ode"]["steps"];
        if (options["ode"]["integration_method"] == "RK2")
            integration_method = 0;
        else if (options["ode"]["integration_method"] == "RK4")
            integration_method = 1;
        else
        {
            std::cerr << "Invaild integration method." << std::endl;
            exit(4);
        }

        for (unsigned int i = 0; i < int(options["ode"]["num_layers"]); i++)
        {
            ode_func_net_linears.push_back(register_module("ode_func_net_linear_" + std::to_string(i), torch::nn::Linear(int(options["gru"]["hidden_size"]), int(options["gru"]["hidden_size"]))));
        }
    }

    nlohmann::json options;
    std::vector<torch::nn::Linear> ode_func_net_linears;
    unsigned int steps;
    unsigned int integration_method;

    torch::Tensor ode_net_forward(const torch::Tensor &y0)
    {
        torch::Tensor y = y0;

        for (auto &linear_layer : ode_func_net_linears)
        {
            y = linear_layer->forward(y);
            y = torch::tanh(y);
        }
        return y;
    }

    torch::Tensor ode_int_RK2(const torch::Tensor &y0, const torch::Tensor &dt)
    {
        torch::Tensor f0 = ode_net_forward(y0);
        torch::Tensor y_mid = y0 + f0 * dt / 2;
        return y0 + ode_net_forward(y_mid) * dt;
    }

    torch::Tensor ode_int_RK4(const torch::Tensor &y0, const torch::Tensor &dt)
    {
        torch::Tensor k1 = ode_net_forward(y0) * dt;
        torch::Tensor k2 = ode_net_forward(y0 + k1 / 2) * dt;
        torch::Tensor k3 = ode_net_forward(y0 + k2 / 2) * dt;
        torch::Tensor k4 = ode_net_forward(y0 + k3) * dt;
        return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6;
    }

    // torch::Tensor forward(torch::Tensor hidden_state, torch::Tensor t0, torch::Tensor t1)
    torch::Tensor forward(torch::Tensor hidden_state, torch::Tensor dt)
    {
        auto integration_dt = (dt / int(steps)).unsqueeze(1);

        for (unsigned int i = 0; i < steps; ++i)
        {
            if (integration_method == 0)
            {
                hidden_state = ode_int_RK2(hidden_state, integration_dt);
            }
            else if (integration_method == 1)
            {
                hidden_state = ode_int_RK4(hidden_state, integration_dt);
            }
            else
            {
                std::cerr << "Invaild integration method." << std::endl;
                exit(4);
            }
        }
        return hidden_state;
    }
};
TORCH_MODULE(ODESolver);