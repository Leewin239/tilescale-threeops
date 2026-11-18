#include <ATen/dlpack.h>

#include <algorithm>
#include <new>

namespace {
void release(DLManagedTensor* value) {
    delete[] value->dl_tensor.shape;
    delete value;
}
}  // namespace

extern "C" DLManagedTensor* tilelang_vmm_view(
    void* ptr, int device, int ndim, const int64_t* dims, uint8_t code, uint8_t bits) {
    auto* view = new (std::nothrow) DLManagedTensor{};
    if (!view) {
        return nullptr;
    }
    auto* shape = new (std::nothrow) int64_t[ndim];
    if (!shape) {
        delete view;
        return nullptr;
    }
    std::copy_n(dims, ndim, shape);
    view->dl_tensor = {ptr, {kDLCUDA, device}, ndim, {code, bits, 1}, shape, nullptr, 0};
    view->deleter = release;
    return view;
}

extern "C" void tilelang_vmm_release(DLManagedTensor* view) {
    release(view);
}
