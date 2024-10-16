
#pragma once

#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <chrono>
#include "Utils.h"

/**
 * This template class is designed to facilitate the swapping of models, datasets, loss functions, optimizers and learning rate schedulers.
 * Use the `make_trainer` function to create an instance of this class.
 */
template <typename ModelType, typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
class Trainer
{
public:
    Trainer(ModelType &model,
            LossFunctionType &loss_function,
            DataLoaderType1 &train_loader,
            DataLoaderType2 &val_loader,
            OptimizerType &optimizer,
            LRSchedulertype &lr_scheduler,
            torch::Device device,
            torch::Dtype dtype)
        : _model(model),
          _loss_function(loss_function),
          _train_loader(train_loader),
          _val_loader(val_loader),
          _optimizer(optimizer),
          _lr_scheduler(lr_scheduler),
          _device(device),
          _dtype(dtype)
    {
        _model->to(dtype);
        _model->to(device);
        // _os.addStream(&std::cout);
        _save_interval = 10;
        _save_name_base = "model";
        _loss_w = 1;
        _train_count = 0;
        _val_count = 0;
    }

    void train()
    {
        checkSettings(false);
        _os << "Epoch,Train Loss,Validation Loss,Wall Time(s)\n";
        double learning_rate = 1;
        for (size_t epoch = 0; epoch < _num_epochs; ++epoch)
        {
            _epoch = epoch;
            auto start = std::chrono::high_resolution_clock::now();

            _train_count = 0;
            _val_count = 0;

            double val_loss = 0.0;
            {
                _model->eval();
                torch::NoGradGuard no_grad;
                for (auto &batch : *_val_loader)
                {
                    torch::Tensor data = batch.data.to(_dtype).to(_device);
                    torch::Tensor target = batch.target.to(_dtype).to(_device);
                    torch::Tensor loss = calculate_loss(data, target);
                    val_loss += loss.item<double>() * _loss_w;
                    _val_count += _loss_w;
                }
            }

            _model->train();
            double train_loss = 0.0;

            for (auto &batch : *_train_loader)
            {
                torch::Tensor data = batch.data.to(_dtype).to(_device);
                torch::Tensor target = batch.target.to(_dtype).to(_device);
                // t_loss: loss with ...
                torch::Tensor t_loss, loss;
                std::tie(t_loss, loss) = calculate_train_loss(data, target);
                _optimizer.zero_grad();
                t_loss.backward();
                _optimizer.step();
                train_loss += loss.item<double>() * _loss_w;
                _train_count += _loss_w;
            }

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            _os << epoch + 1 << "," << train_loss / _train_count << "," << val_loss / _val_count << "," << elapsed.count() << "\n";
            _os.flush();

            if ((epoch + 1) % _save_interval == 0 || (epoch + 1) == _num_epochs)
            {
                _model->to(torch::kCPU);
                std::string model_name = _save_name_base + "-" + std::to_string(epoch + 1) + ".pt";
                torch::save(_model, model_name);
                _model->to(_device);
                std::cout << "Trained model has been saved to " + model_name << std::endl;

                // print learning rate
                for (const auto &param_group : _optimizer.param_groups())
                {
                    learning_rate = param_group.options().get_lr();
                    std::cout << "Current Learning Rate: " << learning_rate << std::endl;
                }
            }

            lr_scheduler_step(val_loss);

            _final_loss = std::make_tuple(train_loss / _train_count, val_loss / _val_count);

            if (learning_rate < 1e-6)
            {
                _model->to(torch::kCPU);
                std::string model_name = _save_name_base + "-" + std::to_string(epoch + 1) + ".pt";
                torch::save(_model, model_name);
                std::cout << "Trained model has been saved to " + model_name << std::endl;
                return;
            }
        }
    }

    void setStreams(std::initializer_list<std::ostream *> streams)
    {
        _os.setStreams(streams);
    }

    void addStream(std::ostream *stream)
    {
        _os.addStream(stream);
    }

    void setNumEpochs(unsigned int num_epochs)
    {
        _num_epochs = num_epochs;
    }

    void setSaveInterval(unsigned int interval)
    {
        _save_interval = interval;
    }

    void setSaveName(std::string name_base)
    {
        _save_name_base = name_base;
    }

    void checkSettings(const bool print_settings = true)
    {
        if (!_num_epochs)
        {
            std::cout << "num_epochs is not set!" << std::endl;
            exit(1);
        }

        if (!print_settings)
        {
            switch (_device.type())
            {
            case torch::kCUDA:
                std::cout << "CUDA available. Training on GPU." << std::endl;
                break;
            case torch::kCPU:
                std::cout << "Training on CPU." << std::endl;
                break;
            default:
                break;
            }
            std::cout << std::endl;
            return;
        }

        std::cout << "num_epochs:\t" << _num_epochs << "\n";
        std::cout << "save_interval:\t" << _save_interval << "\n";
        std::cout << "save_name_base:\t" << _save_name_base << "\n";
        std::cout << std::endl;
    }

    std::tuple<double, double> final_loss()
    {
        return _final_loss;
    }

protected:
    virtual torch::Tensor calculate_loss(torch::Tensor data, torch::Tensor target) = 0;
    virtual std::tuple<torch::Tensor, torch::Tensor> calculate_train_loss(torch::Tensor data, torch::Tensor target)
    {
        torch::Tensor loss = calculate_loss(data, target);
        _loss_w = data.size(0);
        return std::make_tuple(loss, loss);
    }
    virtual void lr_scheduler_step(double /*val_loss*/) = 0;
    ModelType &_model;
    LossFunctionType &_loss_function;
    DataLoaderType1 &_train_loader;
    DataLoaderType2 &_val_loader;
    OptimizerType &_optimizer;
    LRSchedulertype &_lr_scheduler;
    torch::Device _device;
    torch::Dtype _dtype;
    size_t _num_epochs;
    Utils::MultiOstream _os;
    int _save_interval;
    std::string _save_name_base;
    int _loss_w;
    int _train_count;
    int _val_count;
    int _epoch;
    std::tuple<double, double> _final_loss;
};

template <typename ModelType, typename LossFunctionType, typename DataLoaderType1, typename DataLoaderType2, typename OptimizerType, typename LRSchedulertype>
Trainer<ModelType, LossFunctionType, DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>
make_trainer(ModelType &model,
             LossFunctionType &loss_function,
             DataLoaderType1 &train_loader,
             DataLoaderType2 &val_loader,
             OptimizerType &optimizer,
             LRSchedulertype &lr_scheduler,
             torch::Device device,
             torch::Dtype dtype = torch::kFloat64)

{
    return Trainer<ModelType, LossFunctionType,
                   DataLoaderType1, DataLoaderType2, OptimizerType, LRSchedulertype>(model,
                                                                                     loss_function,
                                                                                     train_loader,
                                                                                     val_loader,
                                                                                     optimizer,
                                                                                     lr_scheduler,
                                                                                     device,
                                                                                     dtype);
}
