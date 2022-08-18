// https://www.intel.com/content/www/us/en/develop/documentation/get-started-with-dpcpp-compiler/top.html

#include <CL/sycl.hpp>

int main() {
  // Creating SYCL queue
  cl::sycl::queue Queue;

  // Creating buffer of 4 ints
  cl::sycl::buffer<cl::sycl::cl_int, 1> Buffer(4);

  // Size of index space for kernel
  cl::sycl::range<1> NumOfWorkItems{Buffer.get_count()};

  // Submitting command group to queue
  Queue.submit([&](cl::sycl::handler &cgh) {
    // Getting write only access to the buffer on a device
    auto Accessor = Buffer.get_access<cl::sycl::access::mode::write>(cgh);
    // Executing kernel
    cgh.parallel_for<class FillBuffer>(
        NumOfWorkItems, [=](cl::sycl::id<1> WIid) {
          // Fill buffer with indexes
          Accessor[WIid] = (cl::sycl::cl_int)WIid.get(0);
        });
  });

  // Getting read only access to the buffer on the host
  const auto HostAccessor = Buffer.get_access<cl::sycl::access::mode::read>();

  // Check that the results are correct
  bool MismatchFound = false;
  for (size_t I = 0; I < Buffer.get_count(); ++I) {
    if (HostAccessor[I] != I) {
      std::cout << "The result is incorrect for element: " << I
                << " , expected: " << I << " , got: " << HostAccessor[I]
                << std::endl;
      MismatchFound = true;
    }
  }

  if (!MismatchFound) {
    std::cout << "The results are correct!" << std::endl;
  }

  return MismatchFound;
}
