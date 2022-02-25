#include <ATen/ATen.h>
#include <torch/extension.h>

#include <iostream>
using namespace std;


namespace at {
namespace native {

Tensor n_hot(const Tensor &self, int64_t num_classes) {

  TORCH_CHECK(self.dtype() == kLong,
              "one_hot is only applicable to index tensor.");

  auto shape = self.sizes().vec();

  shape.erase(shape.begin());

  // empty tensor could be converted to one hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return at::empty(shape, self.options());
   }
  }

  // non-empty tensor
  if (self.device().type() != at::kCUDA || self.device().type() != at::kCUDA) {
    // for cuda, rely on device assert thrown by scatter
    TORCH_CHECK(self.min().item().toLong() >= 0,
                "Class values must be non-negative.");
  } else {
    if (self.device().type() != at::kCUDA) {
      // rely on device asserts from scatter to avoid sync here
      TORCH_CHECK(num_classes > self.max().item().toLong(),
                  "Class values must be smaller than num_classes.");
    } else {
      // for cuda, assert that num_classes is at least 1
      TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
    }
  }

  shape.push_back(num_classes);

  Tensor ret = at::zeros(shape);


  for(int i = 0; i < self.size(0); i++) {
    // use the accessor foo_a to get tensor data.
    ret.scatter_(-1, self[i].unsqueeze(-1), 1);
  }

  return ret;
}
} // namespace native
} // namespace at

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("n_hot", &at::native::n_hot, "N Hot Encoding");
} 
