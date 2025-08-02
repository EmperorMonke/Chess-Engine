#include <cmath>
#include <limits>
#include <vector>

class Node {
    double value;
    std::vector<Node> children;

    Node(std::vector<Node> children, double value = std::numeric_limits<double>::quiet_NaN()) {
        if (children.empty()) {
            this->children = children;
        }
        this->value = value;
    }

   double evaluate(Node node) {
        return node.value;
    }

    bool is_terminal (Node node) {
        return std::isnan(node.value);
    }

    std::vector<Node> get_children(Node node) {
        return node.children;
    }

    double alpha_beta_pruning(Node node, int depth, double alpha, double beta, bool maximizing_player) {
        if (depth == 0 || is_terminal(node)) {
            return evaluate(node);
        }

        if (maximizing_player) {
            double max_eval = std::numeric_limits<double>::infinity();
            for (auto child: get_children(node)) {
                double eval = alpha_beta_pruning(child, depth-1, alpha, beta, false);
                max_eval = std::max(max_eval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) {
                    break;
                }
            }
            return max_eval;
        } else {
            double min_eval = std::numeric_limits<double>::infinity();
            for (auto child: get_children(node)) {
                double eval = alpha_beta_pruning(child, depth-1, alpha, beta, true);
                min_eval = std::min(beta, eval);
                if (beta <= alpha) {
                    break;
                }
            }
            return min_eval;
        }
    }
};