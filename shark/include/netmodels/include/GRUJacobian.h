
#pragma once
#include <torch/torch.h>
#include "ODEGRUCell.h"
#include "Tools.h"

struct GRUJacobianImpl : torch::nn::Module
{
    GRUJacobianImpl(nlohmann::json model_options)
        : model_options(model_options),
          ode_gru_cell(model_options)
    {
        input_dim = (model_options["input_output"]["dim"] == 3) ? 6 : 3;

        bool sym = model_options["input_output"]["jacobian_sym"];

        if (sym)
        {
            output_dim = (model_options["input_output"]["dim"] == 3) ? 21 : 6;
        }
        else
        {
            output_dim = (model_options["input_output"]["dim"] == 3) ? 36 : 9;
        }

        output_dim = (model_options["input_output"]["dim"] == 3) ? 36 : 6;
        register_module("ode_gru_cell", ode_gru_cell);
        input_net = register_module("input_net", create_input_sequential());
        output_net = register_module("output_net", create_output_sequential());
    }
    nlohmann::json model_options;
    ODEGRUCell ode_gru_cell;
    unsigned int input_dim;
    unsigned int output_dim;
    torch::nn::Sequential input_net{nullptr};
    torch::nn::Sequential output_net{nullptr};

    torch::nn::Sequential create_input_sequential()
    {
        torch::nn::Sequential seq;

        seq->push_back(torch::nn::Linear(input_dim, int(model_options["gru"]["input_size"])));
        // seq->push_back(torch::nn::BatchNorm1d(int(model_options["gru"]["input_size"])));

        for (unsigned int i = 0; i < int(model_options["input_output"]["input_layers"]) - 1; ++i)
        {
            seq->push_back(torch::nn::LeakyReLU());
            seq->push_back(torch::nn::Linear(int(model_options["gru"]["input_size"]), int(model_options["gru"]["input_size"])));
            // seq->push_back(torch::nn::BatchNorm1d(int(model_options["gru"]["input_size"])));
        }

        return seq;
    }

    torch::nn::Sequential create_output_sequential()
    {
        torch::nn::Sequential seq;

        for (unsigned int i = 0; i < int(model_options["input_output"]["output_layers"]) - 1; ++i)
        {
            seq->push_back(torch::nn::Linear(int(model_options["gru"]["hidden_size"]), int(model_options["gru"]["hidden_size"])));
            // seq->push_back(torch::nn::BatchNorm1d(int(model_options["gru"]["hidden_size"])));
            seq->push_back(torch::nn::LeakyReLU());
        }

        seq->push_back(torch::nn::Linear(int(model_options["gru"]["hidden_size"]), output_dim));
        return seq;
    }

    /**
     *
     * @param timestamped_strain size: {batch, seq, 1 + strain_dim}
     * @param t0 previous timestamp size: {batch, 1}
     * @param hx size: {num_layers, batch_size, hidden_size}
     * @return {  normalized_modulus{batch, seq, 21/6}, h{num_layers, batch_size, hidden_size}  }
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor timestamped_strain, c10::optional<torch::Tensor> t0 = {}, torch::Tensor hx = {})
    {
        std::vector<torch::Tensor> outs;
        int64_t seq_len = timestamped_strain.size(1);
        int64_t batch_size = timestamped_strain.size(0);

        // t size : {batch, seq}
        torch::Tensor t = timestamped_strain.select(-1, 0);
        torch::Tensor dt = Tools::seqence_diff(t, t0, -1);

        // size: {batch, seq, strain_dim}
        torch::Tensor normalized_strain = timestamped_strain.slice(-1, 1);

        // size: {batch, seq, strain_dim}
        auto input = input_net->forward(normalized_strain);

        torch::Tensor h = hx;
        for (int i = 0; i < seq_len; i++)
        {
            torch::Tensor out;

            std::tie(out, h) = ode_gru_cell->forward(input.select(1, i), dt.select(1, i), h);

            outs.push_back(out);
        }

        auto normalized_modulus = output_net->forward(torch::stack(outs, 1));

        return std::make_tuple(normalized_modulus, h);
    }
};
TORCH_MODULE(GRUJacobian);