#include "chess_position_encoding.h"
#include <sstream>
#include <unordered_map>
#include <iostream>

std::vector<std::array<bool, ChessPositionEncoder::BOARD_SIZE * ChessPositionEncoder::BOARD_SIZE>> 
ChessPositionEncoder::encode(
    const std::string& fen, 
    const std::vector<std::string>& previous_positions
) {
    // Reset encoding planes
    encoding_planes.clear();
    encoding_planes.resize(TOTAL_PLANES);

    // Encode current position
    encodePieces(fen, 0);
    encodeRepetitions(previous_positions, 0);
    
    // Encode history positions (up to 8 previous positions)
    const int PLANES_PER_POSITION = 14; // 12 piece planes + 2 repetition planes
    const int MAX_HISTORY = 8;
    
    for (int i = 0; i < MAX_HISTORY; ++i) {
        int history_index = previous_positions.size() - 1 - i;
        int plane_offset = 19 + (i * PLANES_PER_POSITION);
        
        if (history_index >= 0 && history_index < previous_positions.size()) {
            // Valid history position exists
            encodePieces(previous_positions[history_index], plane_offset);
            
            // For repetitions in history, we need to consider up to 8 positions before this one
            // Create a vector containing the current history position and up to 8 positions before it
            std::vector<std::string> historical_context;
            
            // Add the historical position as the target
            historical_context.push_back(previous_positions[history_index]);
            
            // Add up to 8 previous positions (positions before the historical position)
            int positions_to_include = std::min(8, history_index);
            for (int j = history_index - positions_to_include; j < history_index; ++j) {
                if (j >= 0) {
                    historical_context.insert(historical_context.begin(), previous_positions[j]);
                }
            }
            
            // Check for repetitions in this historical context and encode them
            if (historical_context.size() > 1) {  // Need at least 2 positions to have repetitions
                std::vector<std::string> context_for_repetitions;
                for (const auto& pos : historical_context) {
                    context_for_repetitions.push_back(pos);
                    // The last position will be the target position to check for repetitions
                }
                encodeRepetitions(context_for_repetitions, plane_offset);
            }
        }
    }
    
    // Encode additional game state
    encodeCastlingRights(fen);
    encodeColorToMove(fen);

    // Extract move counts from FEN
    std::istringstream fen_stream(fen);
    std::string board, active_color, castling, en_passant, halfmove, fullmove;
    
    std::getline(fen_stream, board, ' ');
    std::getline(fen_stream, active_color, ' ');
    std::getline(fen_stream, castling, ' ');
    std::getline(fen_stream, en_passant, ' ');
    std::getline(fen_stream, halfmove, ' ');
    std::getline(fen_stream, fullmove);

    total_move_count = std::stof(fullmove);
    no_progress_move_count = std::stof(halfmove);

    return encoding_planes;
}

void ChessPositionEncoder::encodePieces(const std::string& fen, int plane_offset) {
    // Piece to plane mapping
    static const std::unordered_map<char, int> white_piece_planes = {
        {'P', WHITE_PAWN_PLANE},
        {'R', WHITE_ROOK_PLANE},
        {'N', WHITE_KNIGHT_PLANE},
        {'B', WHITE_BISHOP_PLANE},
        {'Q', WHITE_QUEEN_PLANE},
        {'K', WHITE_KING_PLANE}
    };

    static const std::unordered_map<char, int> black_piece_planes = {
        {'p', BLACK_PAWN_PLANE},
        {'r', BLACK_ROOK_PLANE},
        {'n', BLACK_KNIGHT_PLANE},
        {'b', BLACK_BISHOP_PLANE},
        {'q', BLACK_QUEEN_PLANE},
        {'k', BLACK_KING_PLANE}
    };

    // Parse board part of FEN
    std::string board_part = fen.substr(0, fen.find(' '));
    int rank = 7, file = 0;

    for (char c : board_part) {
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        
        if (std::isdigit(c)) {
            file += c - '0';
            continue;
        }

        int plane_index = -1;
        if (std::isupper(c)) {
            auto it = white_piece_planes.find(c);
            if (it != white_piece_planes.end()) {
                plane_index = it->second + plane_offset;
            }
        } else {
            auto it = black_piece_planes.find(c);
            if (it != black_piece_planes.end()) {
                plane_index = it->second + plane_offset;
            }
        }

        if (plane_index != -1 && plane_index < TOTAL_PLANES) {
            int board_index = rank * BOARD_SIZE + file;
            encoding_planes[plane_index][board_index] = true;
        }

        file++;
    }
}

void ChessPositionEncoder::encodeRepetitions(const std::vector<std::string>& previous_positions, int plane_offset) {
    // If we have no positions, there are no repetitions
    if (previous_positions.empty()) {
        return;
    }
    
    // Get the position we're checking repetitions for (the last one in the vector)
    std::string target_board = previous_positions.back().substr(0, previous_positions.back().find(' '));
    
    // Count how many times this position appeared in previous positions
    // Look at up to 8 earlier positions
    int repetition_count = 0;
    
    // Calculate how far back to look (up to 8 positions)
    size_t lookback_count = std::min(previous_positions.size() - 1, static_cast<size_t>(8));
    size_t start_idx = previous_positions.size() - 1 - lookback_count;
    
    // Check previous positions for matches with the target position
    for (size_t i = start_idx; i < previous_positions.size() - 1; ++i) {
        std::string board_part = previous_positions[i].substr(0, previous_positions[i].find(' '));
        if (board_part == target_board) {
            repetition_count++;
        }
    }
    
    // Set repetition planes based on how many times we've seen this position before
    if (repetition_count >= 1) {
        int repetition_once_plane = REPETITION_ONCE_PLANE + plane_offset;
        if (repetition_once_plane < TOTAL_PLANES) {
            for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
                encoding_planes[repetition_once_plane][i] = true;
            }
        }
    }

    if (repetition_count >= 2) {
        int repetition_twice_plane = REPETITION_TWICE_PLANE + plane_offset;
        if (repetition_twice_plane < TOTAL_PLANES) {
            for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
                encoding_planes[repetition_twice_plane][i] = true;
            }
        }
    }
}

void ChessPositionEncoder::encodeCastlingRights(const std::string& fen) {
    // Extract castling part from FEN
    std::istringstream fen_stream(fen);
    std::string board, active_color, castling;
    
    std::getline(fen_stream, board, ' ');
    std::getline(fen_stream, active_color, ' ');
    std::getline(fen_stream, castling, ' ');

    // Set castling planes
    for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        encoding_planes[WHITE_KINGSIDE_CASTLE_PLANE][i] = (castling.find('K') != std::string::npos);
        encoding_planes[WHITE_QUEENSIDE_CASTLE_PLANE][i] = (castling.find('Q') != std::string::npos);
        encoding_planes[BLACK_KINGSIDE_CASTLE_PLANE][i] = (castling.find('k') != std::string::npos);
        encoding_planes[BLACK_QUEENSIDE_CASTLE_PLANE][i] = (castling.find('q') != std::string::npos);
    }
}

void ChessPositionEncoder::encodeColorToMove(const std::string& fen) {
    // Extract active color from FEN
    std::istringstream fen_stream(fen);
    std::string board, active_color;
    
    std::getline(fen_stream, board, ' ');
    std::getline(fen_stream, active_color, ' ');

    // Set color to move plane
    bool is_white_to_move = (active_color == "w");
    for (size_t i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
        encoding_planes[COLOR_TO_MOVE_PLANE][i] = is_white_to_move;
    }
}

void ChessPositionEncoder::printEncodingPlanes(int start_plane = 0, int end_plane = -1) const {
    if (end_plane == -1) end_plane = encoding_planes.size();
    
    // Piece plane names (first 12 planes of any position)
    static const std::vector<std::string> piece_plane_names = {
        "White Pawn", "White Rook", "White Knight", "White Bishop", 
        "White Queen", "White King",
        "Black Pawn", "Black Rook", "Black Knight", "Black Bishop", 
        "Black Queen", "Black King"
    };
    
    // Repetition plane names (planes 12-13 of any position)
    static const std::vector<std::string> repetition_plane_names = {
        "Repetition Once", "Repetition Twice"
    };
    
    // Color to move and castling rights (planes 14-18 of current position only)
    static const std::vector<std::string> state_plane_names = {
        "Color to Move",
        "White Kingside Castle", "White Queenside Castle", 
        "Black Kingside Castle", "Black Queenside Castle"
    };

    // Iterate through specified planes
    for (size_t plane_index = start_plane; plane_index < end_plane && plane_index < encoding_planes.size(); ++plane_index) {
        std::string plane_name;
        std::string plane_description;
        
        // Current position - pieces and repetitions (0-13)
        if (plane_index < 12) {
            plane_name = "Current: " + piece_plane_names[plane_index];
            plane_description = "Shows current positions of " + piece_plane_names[plane_index];
        }
        else if (plane_index < 14) {
            plane_name = "Current: " + repetition_plane_names[plane_index - 12];
            plane_description = "Indicates if current position has repeated";
        }
        // Current position - state planes (14-18)
        else if (plane_index < 19) {
            int state_index = plane_index - 14;
            plane_name = "Current: " + state_plane_names[state_index];
            plane_description = "Current game state: " + state_plane_names[state_index];
        }
        // History position planes (19+)
        // Each history position has 14 planes (12 piece planes + 2 repetition planes)
        else {
            int history_index = (plane_index - 19) / 14;
            int plane_type = (plane_index - 19) % 14;
            
            if (plane_type < 12) {
                plane_name = "History " + std::to_string(history_index + 1) + ": " + piece_plane_names[plane_type];
                plane_description = "Shows " + piece_plane_names[plane_type] + " positions from " + 
                                   std::to_string(history_index + 1) + " moves ago";
            } else {
                plane_name = "History " + std::to_string(history_index + 1) + ": " + 
                            repetition_plane_names[plane_type - 12];
                plane_description = "Indicates if position from " + std::to_string(history_index + 1) + 
                                   " moves ago has repeated";
            }
        }

        // Print plane header with description
        std::cout << "Plane " << plane_index << ": " << plane_name << std::endl;
        std::cout << "Description: " << plane_description << std::endl;

        // Print 8x8 grid for each plane
        for (int rank = 7; rank >= 0; --rank) {
            for (int file = 0; file < 8; ++file) {
                int index = rank * 8 + file;
                std::cout << (encoding_planes[plane_index][index] ? "1 " : "0 ");
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Print additional scalar inputs
    std::cout << "Additional Scalar Inputs:" << std::endl;
    std::cout << "Total Move Count: " << total_move_count << std::endl;
    std::cout << "No Progress Move Count: " << no_progress_move_count << std::endl;
}

int main() {
    ChessPositionEncoder encoder;

    // Starting position FEN
    std::string starting_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 5 6";
    
    // Empty previous positions for the starting position
    std::vector<std::string> previous_positions = {"rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1", "r1bqkbnr/pppppppp/2n5/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2", "r1bqkbnr/pppppppp/2n5/8/8/2N2N2/PPPPPPPP/R1BQKB1R b KQkq - 3 2", "r1bqkb1r/pppppppp/2n2n2/8/8/2N2N2/PPPPPPPP/R1BQKB1R w KQkq - 4 3", "r1bqkb1r/pppppppp/2n2n2/8/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq e3 0 3", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq e6 0 4", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 1 4", "r1bqk2r/ppppbppp/2n2n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 2 5", "r1bqk2r/ppppbppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 3 5", "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 6"};

    // Encode the position
    auto encoding = encoder.encode(starting_fen, previous_positions);

    // Print out the encoding planes
    encoder.printEncodingPlanes(0, 39);

    return 0;
}
