#include "TimestampedSequenceDataset.h"

TimestampedSequenceDataset::TimestampedSequenceDataset(const torch::Tensor &timestamps,
                                                       const torch::Tensor &inputs,
                                                       const torch::Tensor &outputs) : _outputs(outputs)
{
    _timestamped_inputs = torch::cat({timestamps.unsqueeze(-1), inputs}, -1);
    _size = timestamps.size(0);
}

torch::optional<size_t> TimestampedSequenceDataset::size() const
{
    return _size;
}

torch::data::Example<> TimestampedSequenceDataset::get(size_t index)
{
    return {_timestamped_inputs[index], _outputs[index]};
}
