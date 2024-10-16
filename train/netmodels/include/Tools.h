#pragma once
#include <torch/torch.h>

namespace Tools
{
    /**
     * @param t  {..., seq, ...}
     * @param prepend {..., 1, ...}
     *
     */
    inline torch::Tensor
    seqence_diff(const torch::Tensor t, c10::optional<torch::Tensor> prepend = {}, int seq_dim = -2)
    {
        if (!prepend.has_value())
        {
            return t.diff(1, seq_dim, torch::zeros_like(t.slice(seq_dim, 0, 1), t.options()));
        }

        return t.diff(1, seq_dim, prepend);
    }

    /**
     * @param t  {..., seq, ...}
     * @param prepend {..., 1, ...}
     *
     */
    inline torch::Tensor
    seqence_accum(const torch::Tensor t, c10::optional<torch::Tensor> prepend = {}, int seq_dim = -2)
    {
        if (prepend.has_value())
        {
            torch::Tensor rt = t;
            rt.slice(seq_dim, 0, 1) += prepend.value();
            return rt.cumsum(seq_dim);
        }

        return t.cumsum(seq_dim);
    }

    /**
     * usage:
     *
     * auto map = symmetric_stiffness_mapping_matrix(C.options());
     * stress = torch::matmul(C, map) * strain;
     */
    inline torch::Tensor
    symmetric_stiffness_mapping_matrix(const torch::Tensor stiffness)
    {
        if (stiffness.size(-1) == 6)
        {
            return torch::tensor(
                {
                    {1, 0, 0}, // 1
                    {1, 1, 0}, // 2
                    {2, 0, 1}, // 3
                    {0, 1, 0}, // 4
                    {0, 2, 1}, // 5
                    {0, 0, 2}  // 6
                },
                stiffness.options());
        }

        if (stiffness.size(-1) == 9)
        {
            return torch::tensor(
                {
                    {1, 0, 0},
                    {1, 0, 0},
                    {2, 0, 0},
                    {0, 1, 0},
                    {0, 1, 0},
                    {0, 2, 0},
                    {0, 0, 1},
                    {0, 0, 1},
                    {0, 0, 2},
                },
                stiffness.options());
        }

        if (stiffness.size(-1) == 36)
        {
            return torch::tensor(
                {{1, 0, 0, 0, 0, 0},
                 {1, 0, 0, 0, 0, 0},
                 {1, 0, 0, 0, 0, 0},
                 {2, 0, 0, 0, 0, 0},
                 {2, 0, 0, 0, 0, 0},
                 {2, 0, 0, 0, 0, 0},

                 {0, 1, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0, 0},
                 {0, 1, 0, 0, 0, 0},
                 {0, 2, 0, 0, 0, 0},
                 {0, 2, 0, 0, 0, 0},
                 {0, 2, 0, 0, 0, 0},

                 {0, 0, 1, 0, 0, 0},
                 {0, 0, 1, 0, 0, 0},
                 {0, 0, 1, 0, 0, 0},
                 {0, 0, 2, 0, 0, 0},
                 {0, 0, 2, 0, 0, 0},
                 {0, 0, 2, 0, 0, 0},

                 {0, 0, 0, 1, 0, 0},
                 {0, 0, 0, 1, 0, 0},
                 {0, 0, 0, 1, 0, 0},
                 {0, 0, 0, 2, 0, 0},
                 {0, 0, 0, 2, 0, 0},
                 {0, 0, 0, 2, 0, 0},

                 {0, 0, 0, 0, 1, 0},
                 {0, 0, 0, 0, 1, 0},
                 {0, 0, 0, 0, 1, 0},
                 {0, 0, 0, 0, 2, 0},
                 {0, 0, 0, 0, 2, 0},
                 {0, 0, 0, 0, 2, 0},

                 {0, 0, 0, 0, 0, 1},
                 {0, 0, 0, 0, 0, 1},
                 {0, 0, 0, 0, 0, 1},
                 {0, 0, 0, 0, 0, 2},
                 {0, 0, 0, 0, 0, 2},
                 {0, 0, 0, 0, 0, 2}},
                stiffness.options());
        }

        if (stiffness.size(-1) == 21)
        {
            return torch::tensor(
                {
                    {1, 0, 0, 0, 0, 0}, // 1
                    {1, 1, 0, 0, 0, 0}, // 2
                    {1, 0, 1, 0, 0, 0}, // 3
                    {2, 0, 0, 1, 0, 0}, // 4
                    {2, 0, 0, 0, 1, 0}, // 5
                    {2, 0, 0, 0, 0, 1}, // 6
                    {0, 1, 0, 0, 0, 0}, // 7
                    {0, 1, 1, 0, 0, 0}, // 8
                    {0, 2, 0, 1, 0, 0}, // 9
                    {0, 2, 0, 0, 1, 0}, // 10
                    {0, 2, 0, 0, 0, 1}, // 11
                    {0, 0, 1, 0, 0, 0}, // 12
                    {0, 0, 2, 1, 0, 0}, // 13
                    {0, 0, 2, 0, 1, 0}, // 14
                    {0, 0, 2, 0, 0, 1}, // 15
                    {0, 0, 0, 2, 0, 0}, // 16
                    {0, 0, 0, 2, 2, 0}, // 17
                    {0, 0, 0, 2, 0, 2}, // 18
                    {0, 0, 0, 0, 2, 0}, // 19
                    {0, 0, 0, 0, 2, 2}, // 20
                    {0, 0, 0, 0, 0, 2}, // 21
                },
                stiffness.options());
        }

        std::cout << " Unsupported stiffness dim " << stiffness.size(-1) << std::endl;
        exit(1);
        return torch::Tensor();
    }

    inline torch::Tensor
    calculate_delta_stress(torch::Tensor stiffness, const torch::Tensor strain, c10::optional<torch::Tensor> strain0 = {})
    {
        auto dstrain = seqence_diff(strain, strain0);
        auto dstress = torch::matmul(stiffness, symmetric_stiffness_mapping_matrix(stiffness)) * dstrain;
        return dstress;
    }

    inline torch::Tensor
    calculate_symmetric_stress(torch::Tensor stiffness,
                               const torch::Tensor strain,
                               c10::optional<torch::Tensor> strain0 = {},
                               c10::optional<torch::Tensor> stress0 = {})
    {
        auto dstress = calculate_delta_stress(stiffness, strain, strain0);
        auto stress = seqence_accum(dstress, stress0);
        return stress;
    }
}