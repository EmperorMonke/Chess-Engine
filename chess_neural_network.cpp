#include "chess_neural_network.h"

// Constructor
ChessNeuralNetwork::ChessNeuralNetwork(int num_residual_blocks) {
    // Input features: 117 planes (from the encoder) of 8x8
    // Initial convolution
    input_conv = register_module("input_conv", 
        torch::nn::Conv2d(torch::nn::Conv2dOptions(117, 256, 3)
            .stride(1)
            .padding(1)
            .bias(false)));
    
    // Create residual blocks - repeated #blocks times as shown in diagram
    for (int i = 0; i < num_residual_blocks; ++i) {
        auto block = torch::nn::Sequential();
        
        // First convolution in residual block
        block->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256, 256, 3)
                .stride(1)
                .padding(1)
                .bias(false)));
        block->push_back(torch::nn::BatchNorm2d(256));
        block->push_back(torch::nn::ReLU());
        
        // Second convolution in residual block
        block->push_back(torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256, 256, 3)
                .stride(1)
                .padding(1)
                .bias(false)));
        block->push_back(torch::nn::BatchNorm2d(256));
        
        // Squeeze and excitation as shown in diagram
        auto se = torch::nn::Sequential();
        se->push_back(torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        se->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 16, 1)));
        se->push_back(torch::nn::ReLU());
        se->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 256, 1)));
        se->push_back(torch::nn::Sigmoid());
        block->push_back(se);
        
        // Register and add the block
        residual_blocks.push_back(register_module("residual_block_" + std::to_string(i), block));
    }
    
    // Policy head - outputs 73x8x8 as shown (AlphaZero style)
    policy_conv1 = register_module("policy_conv1",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3)
            .stride(1)
            .padding(1)
            .bias(false)));
            
    policy_conv2 = register_module("policy_conv2",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 73, 1)
            .stride(1)
            .padding(0)));
    
    // Value head - outputs win/draw/loss probabilities
    value_conv = register_module("value_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 32, 1)
            .stride(1)
            .padding(0)
            .bias(false)));
            
    value_fc1 = register_module("value_fc1", 
        torch::nn::Linear(32 * 8 * 8, 128));
        
    value_fc2 = register_module("value_fc2", 
        torch::nn::Linear(128, 3));  // 3 outputs: win, draw, loss
    
    // Moves-left head - outputs number of moves left estimate
    moves_left_conv = register_module("moves_left_conv",
        torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 8, 1)
            .stride(1)
            .padding(0)
            .bias(false)));
            
    moves_left_fc1 = register_module("moves_left_fc1", 
        torch::nn::Linear(8 * 8 * 8, 128));
        
    moves_left_fc2 = register_module("moves_left_fc2", 
        torch::nn::Linear(128, 1));  // Single output: moves left
}

// Forward pass
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
ChessNeuralNetwork::forward(torch::Tensor x, torch::Tensor scalar_input) {
    // x shape: [batch_size, 117, 8, 8]
    // scalar_input shape: [batch_size, 2] - total move count and no progress move count
    
    // Initial convolution
    auto out = torch::relu(input_conv->forward(x));
    
    // Residual blocks with skip connections
    for (auto& block : residual_blocks) {
        auto residual = out;
        out = block->forward(out);
        out = torch::relu(out + residual);  // Skip connection
    }
    
    // Policy head
    auto policy_out = torch::relu(policy_conv1->forward(out));
    policy_out = policy_conv2->forward(policy_out);
    // Shape: [batch_size, 73, 8, 8] - these are the action probabilities
    
    // Value head
    auto value_out = torch::relu(value_conv->forward(out));
    value_out = value_out.view({value_out.size(0), -1});  // Flatten
    value_out = torch::relu(value_fc1->forward(value_out));
    value_out = torch::tanh(value_fc2->forward(value_out));
    value_out = torch::softmax(value_out, 1);  // Softmax over win/draw/loss
    
    // Moves-left head
    auto moves_left_out = torch::relu(moves_left_conv->forward(out));
    moves_left_out = moves_left_out.view({moves_left_out.size(0), -1});  // Flatten
    moves_left_out = torch::relu(moves_left_fc1->forward(moves_left_out));
    moves_left_out = moves_left_fc2->forward(moves_left_out);
    // Single value output for estimated moves left
    
    return {policy_out, value_out, moves_left_out};
}