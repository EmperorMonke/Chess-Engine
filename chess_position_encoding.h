#ifndef CHESS_POSITION_ENCODING_H
#define CHESS_POSITION_ENCODING_H

#include <vector>
#include <string>
#include <array>
#include <cstdint>

class ChessPositionEncoder {
public:
    // Constant dimensions based on AlphaZero encoding
    static constexpr int BOARD_SIZE = 8;
    static constexpr int PIECE_TYPES = 6;
    static constexpr int TOTAL_PLANES = 117;

    // Getter methods for scalar values
    float getTotalMoveCount() const { return total_move_count; }
    float getNoProgressMoveCount() const { return no_progress_move_count; }

    // Plane indices for different encoding components
    enum PlaneIndices {
        WHITE_PAWN_PLANE = 0,
        WHITE_ROOK_PLANE,
        WHITE_KNIGHT_PLANE,
        WHITE_BISHOP_PLANE,
        WHITE_QUEEN_PLANE,
        WHITE_KING_PLANE,
        BLACK_PAWN_PLANE = 6,
        BLACK_ROOK_PLANE,
        BLACK_KNIGHT_PLANE,
        BLACK_BISHOP_PLANE,
        BLACK_QUEEN_PLANE,
        BLACK_KING_PLANE,
        REPETITION_ONCE_PLANE,
        REPETITION_TWICE_PLANE,
        COLOR_TO_MOVE_PLANE,
        WHITE_KINGSIDE_CASTLE_PLANE,
        WHITE_QUEENSIDE_CASTLE_PLANE,
        BLACK_KINGSIDE_CASTLE_PLANE,
        BLACK_QUEENSIDE_CASTLE_PLANE
    };

    // Main encoding method
    std::vector<std::array<bool, BOARD_SIZE * BOARD_SIZE>> encode(
        const std::string& fen, 
        const std::vector<std::string>& previous_positions
    );

private:
    // Helper methods for encoding different aspects of the position
    void encodePieces(const std::string& fen, int plane_offset);
    void encodeRepetitions(const std::vector<std::string>& previous_positions, int plane_offset);
    void encodeCastlingRights(const std::string& fen);
    void encodeColorToMove(const std::string& fen);
    public: void printEncodingPlanes(int start_plane, int end_plane) const;

    protected:
    std::vector<std::array<bool, BOARD_SIZE * BOARD_SIZE>> encoding_planes;
    float total_move_count;
    float no_progress_move_count;
};

#endif // CHESS_POSITION_ENCODING_H