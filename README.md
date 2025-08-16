A chess engine written in [C++], currently in development. This project aims to create a lightweight, efficient, and reasonably strong chess engine via the [UCI] protocol.

## 🚧 Project Status

**Under development** – core functionality is being implemented and optimized. Not yet ready for competitive play or full compliance testing.

## 📦 Features (Planned / In Progress)

- [x] Move generation
- [ ] Perft testing and debugging
- [ ] Evaluation function
- [ ] Minimax / Negamax search
- [x] Alpha-beta pruning
- [x] UCI protocol support
- [ ] Transposition table (Zobrist hashing)
- [ ] Opening book
- [ ] Endgame heuristics / Tablebase support

## 🔧 Built With

- Language: C++
- Tools: CMake, Git
- Standards: UCI

## 🧪 Testing

You can run perft tests using:

```bash
python tests/perft.py