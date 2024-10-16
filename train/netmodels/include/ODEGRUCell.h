#pragma once
#include <torch/torch.h>
#include "ODESolver.h"
#if __has_include(<nlohmann/json.hpp>)
#include <nlohmann/json.hpp>
#elif __has_include("nlohmann/json.h")
#include "nlohmann/json.h"
#else
#error "nlohmann/json header file not found"
#endif

struct ODEGRUCellImpl : torch::nn::Module
{
    // input_size: input size of gru cell
    // hidden_size: hidden layer size of gru cell
    ODEGRUCellImpl(nlohmann::json options)
        : options(options),
          initial_hidden_state(torch::zeros({int(options["gru"]["num_layers"]), 1, int(options["gru"]["hidden_size"])}))
    {
        register_parameter("initial_hidden_state", initial_hidden_state);
        for (unsigned int i = 0; i < initial_hidden_state.size(0); ++i)
        {
            auto gru_cell = register_module("gru_cell_" + std::to_string(i),
                                            torch::nn::GRUCell(i == 0 ? int(options["gru"]["input_size"]) : int(options["gru"]["hidden_size"]),
                                                               int(options["gru"]["hidden_size"])));

            torch::nn::init::orthogonal_(gru_cell->weight_ih);
            torch::nn::init::orthogonal_(gru_cell->weight_hh);
            torch::nn::init::zeros_(gru_cell->bias_ih);
            torch::nn::init::zeros_(gru_cell->bias_hh);
            torch::nn::init::xavier_uniform_(initial_hidden_state[i]);

            gru_cells.push_back(gru_cell);

            auto ode_solver = register_module("ode_solver_" + std::to_string(i), ODESolver(options));
            ode_solvers.push_back(ode_solver);
        }
    }

    nlohmann::json options;
    torch::Tensor initial_hidden_state;
    std::vector<ODESolver> ode_solvers;
    std::vector<torch::nn::GRUCell> gru_cells;

    /**
     *
     * @param input size: {batch, input_size}
     * @param dt size: {batch}
     * @param hidden_states size: {num_layers, batch, hidden_size}
     * @return output size: {batch, hidden_size}
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor input, torch::Tensor dt, torch::Tensor hidden_states = {})
    {
        auto batch_size = input.size(0);
        if (!hidden_states.defined())
        {
            hidden_states = initial_hidden_state.repeat({1, batch_size, 1});
        }

        torch::Tensor new_hidden_states = torch::zeros_like(hidden_states);
        auto output = input;

        for (unsigned int i = 0; i < initial_hidden_state.size(0); ++i)
        {
            auto hidden_state_ = ode_solvers[i]->forward(hidden_states[i], dt);
            output = gru_cells[i]->forward(output, hidden_state_);
            new_hidden_states[i] = output;
        }
        return std::make_tuple(output, new_hidden_states);
    }
};
TORCH_MODULE(ODEGRUCell);