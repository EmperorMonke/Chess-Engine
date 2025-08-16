#include <limits>
#include <map>
#include "chess.hpp"

using namespace chess;

double evaluate(const Board& board) {
    if (board.isGameOver().second == chess::GameResult::WIN) {
        return (board.sideToMove() == Color::WHITE) 
            ? 10000.0 
            : -10000.0;
    }
    if (board.isGameOver().second == GameResult::DRAW) {
        return 0.0;
    }

    double material = 0.0;
    const std::map<PieceType, double> values = {
        {PieceType::PAWN, 1.0},
        {PieceType::KNIGHT, 3.0},
        {PieceType::BISHOP, 3.1},
        {PieceType::ROOK, 5.0},
        {PieceType::QUEEN, 9.0}
    };

    for (const auto& [piece, val] : values) {
        material += val * board.pieces(piece, Color::WHITE).count();
        material -= val * board.pieces(piece, Color::BLACK).count();
    }
    return material;
}

double alpha_beta(Board& board, int depth, double alpha, double beta) {
    if (depth == 0 || !(board.isGameOver().second == GameResult::NONE)) {
        return evaluate(board);
    }

    Movelist moves;
    movegen::legalmoves(moves, board);

    if (board.sideToMove() == Color::WHITE) {
        double max_eval = -std::numeric_limits<double>::infinity();
        for (const auto& move : moves) {
            board.makeMove(move);
            double eval = alpha_beta(board, depth - 1, alpha, beta);
            board.unmakeMove(move);
            max_eval = std::max(max_eval, eval);
            alpha = std::max(alpha, eval);
            if (beta <= alpha) break;
        }
        return max_eval;
    } else {
        double min_eval = std::numeric_limits<double>::infinity();
        for (const auto& move : moves) {
            board.makeMove(move);
            double eval = alpha_beta(board, depth - 1, alpha, beta);
            board.unmakeMove(move);
            min_eval = std::min(min_eval, eval);
            beta = std::min(beta, eval);
            if (beta <= alpha) break;
        }
        return min_eval;
    }
}

Move get_best_move(Board& board, int depth) {
    Movelist moves;
    movegen::legalmoves(moves, board);

    if (moves.empty()) {
        return Move::NO_MOVE;
    }

    double alpha = -std::numeric_limits<double>::infinity();
    double beta = std::numeric_limits<double>::infinity();
    double best_value;
    Move best_move = moves[0];

    if (board.sideToMove() == Color::WHITE) {
        best_value = -std::numeric_limits<double>::infinity();
        for (const auto& move : moves) {
            board.makeMove(move);
            double value = alpha_beta(board, depth - 1, alpha, beta);
            board.unmakeMove(move);
            if (value > best_value) {
                best_value = value;
                best_move = move;
            }
            alpha = std::max(alpha, best_value);
            if (beta <= alpha) break;
        }
    } else {
        best_value = std::numeric_limits<double>::infinity();
        for (const auto& move : moves) {
            board.makeMove(move);
            double value = alpha_beta(board, depth - 1, alpha, beta);
            board.unmakeMove(move);
            if (value < best_value) {
                best_value = value;
                best_move = move;
            }
            beta = std::min(beta, best_value);
            if (beta <= alpha) break;
        }
    }
    return best_move;
}

int main() {
    Board board = Board();
    int depth = 6;
    
    while (board.isGameOver().second == GameResult::NONE) {
        std::cout << "Current Board:\n" << board << "\n";
        
        if (board.sideToMove() == Color::WHITE) {
            // Human player (White)
            std::string move_str;
            std::cout << "Enter your move (e.g. e2e4): ";
            std::cin >> move_str;
            
            try {
                Move move = uci::uciToMove(board, move_str);
                board.makeMove(move);
            } catch (const std::exception& e) {
                std::cout << "Invalid move! Try again.\n";
                continue;
            }
        } else {
            // Engine player (Black)
            Move best_move = get_best_move(board, depth);
            std::cout << "Engine plays: " << uci::moveToUci(best_move) << "\n";
            board.makeMove(best_move);
        }
    }
    
    std::cout << "Final Board:\n" << board << "\n";
    return 0;
}