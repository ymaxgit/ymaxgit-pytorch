#pragma once

#include <string>

#include "lazy_tensor_core/csrc/device.h"
#include "torch/csrc/lazy/core/ir.h"

namespace torch_lazy_tensors {
namespace ir {

class DumpUtil {
 public:
  static std::string ToDot(const std::vector<const torch::lazy::Node *> &nodes);

  static std::string PostOrderToDot(
      const std::vector<const torch::lazy::Node *> &post_order,
      const std::vector<const torch::lazy::Node *> &roots);

  static std::string ToText(
      const std::vector<const torch::lazy::Node *> &nodes);

  static std::string PostOrderToText(
      const std::vector<const torch::lazy::Node *> &post_order,
      const std::vector<const torch::lazy::Node *> &roots);

  static std::string ToBackend(lazy_tensors::Span<const torch::lazy::Value> values,
                               const Device& device);
};

}  // namespace ir
}  // namespace torch_lazy_tensors
