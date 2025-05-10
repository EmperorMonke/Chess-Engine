#ifndef CHESS_NEURAL_NETWORK_H
#define CHESS_NEURAL_NETWORK_H

#include "chess_position_encoding.h"
#include <torch/torch.h>

class ChessNeuralNetwork : public torch::nn::Module {
public:
    // Constructor
    ChessNeuralNetwork(int num_residual_blocks = 19);

    // Forward pass
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor scalar_input);

private:
    // Input processing
    torch::nn::Conv2d input_conv{nullptr};
    
    // Residual blocks
    std::vector<torch::nn::Sequential> residual_blocks;
    
    // Policy head
    torch::nn::Conv2d policy_conv1{nullptr};
    torch::nn::Conv2d policy_conv2{nullptr};
    
    // Value head
    torch::nn::Conv2d value_conv{nullptr};
    torch::nn::Linear value_fc1{nullptr};
    torch::nn::Linear value_fc2{nullptr};
    
    // Moves-left head
    torch::nn::Conv2d moves_left_conv{nullptr};
    torch::nn::Linear moves_left_fc1{nullptr};
    torch::nn::Linear moves_left_fc2{nullptr};
};

#endif // CHESS_NEURAL_NETWORK_H