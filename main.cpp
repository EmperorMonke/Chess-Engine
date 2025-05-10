#include "chess_position_encoding.h"
#include "chess_neural_network.h"
#include <iostream>
#include <vector>


int main() {
    // Initialize the encoder
    ChessPositionEncoder encoder;
    
    // Initialize the neural network
    ChessNeuralNetwork network(19);  // 19 residual blocks as shown in diagram
    
    // Example FEN for the starting position
    std::string starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    
    // Previous positions (empty for starting position)
    std::vector<std::string> previous_positions;
    
    // Encode the position
    auto encoding_planes = encoder.encode(starting_fen, previous_positions);
    
    // Convert encoding to torch tensor
    std::vector<float> flat_encoding;
    for (const auto& plane : encoding_planes) {
        for (const auto& cell : plane) {
            flat_encoding.push_back(cell ? 1.0f : 0.0f);
        }
    }
    
    // Create input tensor with batch size 1
    torch::Tensor input_tensor = torch::from_blob(flat_encoding.data(), 
                                                 {1, 117, 8, 8},  // batch, channels, height, width
                                                 torch::kFloat);
    
    // Create scalar input tensor (total move count and no progress move count)
    torch::Tensor scalar_input = torch::tensor({{encoder.getTotalMoveCount(), 
                                               encoder.getNoProgressMoveCount()}}, 
                                             torch::kFloat);
    
    // Forward pass through the network
    torch::NoGradGuard no_grad;  // Disable gradient computation for inference
    auto [policy, value, moves_left] = network.forward(input_tensor, scalar_input);
    
    // Print output shapes and values
    std::cout << "Policy output shape: " << policy.sizes() << std::endl;
    std::cout << "Value output shape: " << value.sizes() << std::endl;
    std::cout << "Moves left output shape: " << moves_left.sizes() << std::endl;
    
    // Print win probability from value head
    std::cout << "Win probability: " << value[0][0].item<float>() << std::endl;
    std::cout << "Draw probability: " << value[0][1].item<float>() << std::endl;
    std::cout << "Loss probability: " << value[0][2].item<float>() << std::endl;
    
    // Print estimated moves left
    std::cout << "Estimated moves left: " << moves_left[0][0].item<float>() << std::endl;
    
    return 0;
}