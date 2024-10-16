#pragma once
#include <torch/torch.h>

class TimestampedSequenceDataset : public torch::data::Dataset<TimestampedSequenceDataset>
{
public:
    /**
     * Constructor
     *
     * @param timestamps timestamps size: {batch, seq}
     * @param inputs inputs size: {batch, seq, in_dim}
     * @param outputs outputs size: {batch, seq, out_dim}
     */
    TimestampedSequenceDataset(const torch::Tensor &timestamps,
                               const torch::Tensor &inputs,
                               const torch::Tensor &outputs);

    torch::optional<size_t> size() const override;

    /**
     * Returns a single sample from the dataset
     *
     * {timestamped_input, output}
     * timestamped_input size: {seq, 1+in_dim}
     * output size:{seq, out_dim}
     */
    torch::data::Example<> get(size_t index) override;

private:
    torch::Tensor _outputs;
    torch::Tensor _timestamped_inputs;
    size_t _size;
};
