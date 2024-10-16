#pragma once
#include <torch/torch.h>
#include "GRUJacobian.h"
#include "Tools.h"

struct GRUaStressImpl : torch::nn::Module
{
    GRUaStressImpl(nlohmann::json model_options)
        : model_options(model_options),
          gru_jacobian(model_options)
    {
        register_module("gru_jacobian", gru_jacobian);
    }
    nlohmann::json model_options;
    GRUJacobian gru_jacobian;

    /**
     *
     * @param timestamped_strain size: {batch, seq, 1 + strain_dim}
     * @param stress0 previous stress size: {batch, 1, stress_dim}
     * @param stress0 previous stress size: {batch, 1, stress_dim}
     * @param t0 previous timestamp size: {batch, 1}
     * @param hx size: {num_layers, batch_size, hidden_size}
     *
     * @return {  cat{stress{batch, seq, stress_dim}, normalized_modulus{batch, seq, 21/6}}, h  }
     */
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor timestamped_strain,
                                                     c10::optional<torch::Tensor> strain0 = {},
                                                     c10::optional<torch::Tensor> stress0 = {},
                                                     c10::optional<torch::Tensor> t0 = {},
                                                     torch::Tensor hx = {})
    {
        torch::Tensor normalized_strain = timestamped_strain.slice(-1, 1);

        auto batch_size = normalized_strain.size(0);
        auto seq_len = normalized_strain.size(1);
        auto strain_dim = normalized_strain.size(-1);

        torch::Tensor h, normalized_modulus;
        std::tie(normalized_modulus, h) = gru_jacobian->forward(timestamped_strain, t0, hx);
        torch::Tensor d_normalized_strain_matrix = Tools::seqence_diff(normalized_strain, strain0).unsqueeze(-1);

        torch::Tensor modulus_matrix = normalized_modulus.view({batch_size, seq_len, strain_dim, strain_dim});
        torch::Tensor d_stress = torch::matmul(modulus_matrix, d_normalized_strain_matrix).squeeze(-1);
        torch::Tensor stress = Tools::seqence_accum(d_stress, stress0);

        // torch::Tensor stress = Tools::calculate_symmetric_stress(normalized_modulus, normalized_strain, strain0, stress0);
        return std::make_tuple(torch::cat({stress, normalized_modulus}, -1), h);
    }
};
TORCH_MODULE(GRUaStress);