#pragma once
#include <torch/torch.h>

class ReduceLROnPlateau
{
public:
    ReduceLROnPlateau(torch::optim::Optimizer &optimizer, double factor = 0.1, int patience = 10,
                      double relative_threshold = 1e-3, int cooldown = 5)
        : _optimizer(optimizer),
          _factor(factor),
          _patience(patience),
          _relative_threshold(relative_threshold),
          _cooldown(cooldown),
          _best_loss(std::numeric_limits<double>::infinity()),
          _epochs_since_last_change(0)
    {
    }

    void step(double val_loss)
    {
        if (_cooldown_counter > 0)
        {
            --_cooldown_counter;
            return;
        }

        if (val_loss < _best_loss * (1 - _relative_threshold))
        {
            _best_loss = val_loss;
            _epochs_since_last_change = 0;
        }
        else
        {
            ++_epochs_since_last_change;
            if (_epochs_since_last_change >= _patience)
            {
                for (auto &group : _optimizer.param_groups())
                {
                    group.options().set_lr(group.options().get_lr() * _factor);
                }
                _epochs_since_last_change = 0;
                _cooldown_counter = _cooldown;
            }
        }
    }

private:
    torch::optim::Optimizer &_optimizer;
    double _factor;
    int _patience;
    double _relative_threshold;
    int _cooldown;
    double _best_loss;
    int _epochs_since_last_change;
    int _cooldown_counter;
};
