/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#include <iostream>

#include <cuda.h>
#include <nvrtc.h>

#define NVRTC_SAFE_CALL(x)                                                                      \
  do                                                                                            \
  {                                                                                             \
    nvrtcResult result = x;                                                                     \
    if (result != NVRTC_SUCCESS)                                                                \
    {                                                                                           \
      std::cerr << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

const char* trie =
  R"xxx(

#include <uscl/std/cstddef>
#include <uscl/std/cstdint>
#include <uscl/std/atomic>

template<class T> static constexpr T min(T a, T b) { return a < b ? a : b; }

struct trie {
    struct ref {
        cuda::std::atomic<trie*> ptr = LIBCUDACXX_ATOMIC_VAR_INIT(nullptr);
        // the flag will protect against multiple pointer updates
        cuda::std::atomic_flag flag = LIBCUDACXX_ATOMIC_FLAG_INIT;
    } next[26];
    cuda::std::atomic<int> count = LIBCUDACXX_ATOMIC_VAR_INIT(0);
};
__host__ __device__
int index_of(char c) {
    if(c >= 'a' && c <= 'z') return c - 'a';
    if(c >= 'A' && c <= 'Z') return c - 'A';
    return -1;
};
__host__ __device__
void make_trie(/* trie to insert word counts into */ trie& root,
               /* bump allocator to get new nodes*/ cuda::std::atomic<trie*>& bump,
               /* input */ const char* begin, const char* end,
               /* thread this invocation is for */ unsigned index,
               /* how many threads there are */ unsigned domain) {

    auto const size = end - begin;
    auto const stride = (size / domain + 1);

    auto off = min(size, stride * index);
    auto const last = min(size, off + stride);

    for(char c = begin[off]; off < size && off != last && c != 0 && index_of(c) != -1; ++off, c = begin[off]);
    for(char c = begin[off]; off < size && off != last && c != 0 && index_of(c) == -1; ++off, c = begin[off]);

    trie *n = &root;
    for(char c = begin[off]; ; ++off, c = begin[off]) {
        auto const index = off >= size ? -1 : index_of(c);
        if(index == -1) {
            if(n != &root) {
                n->count.fetch_add(1, cuda::std::memory_order_relaxed);
                n = &root;
            }
            //end of last word?
            if(off >= size || off > last)
                break;
            else
                continue;
        }
        if(n->next[index].ptr.load(cuda::std::memory_order_acquire) == nullptr) {
            if(n->next[index].flag.test_and_set(cuda::std::memory_order_relaxed))
                while(n->next[index].ptr.load(cuda::std::memory_order_acquire) == nullptr);
            else {
                auto next = bump.fetch_add(1, cuda::std::memory_order_relaxed);
                n->next[index].ptr.store(next, cuda::std::memory_order_release);
            }
        }
        n = n->next[index].ptr.load(cuda::std::memory_order_relaxed);
    }
}

__global__ // __launch_bounds__(1024, 1)
void call_make_trie(trie* t, cuda::std::atomic<trie*>* bump, const char* begin, const char* end) {

    auto const index = blockDim.x * blockIdx.x + threadIdx.x;
    auto const domain = gridDim.x * blockDim.x;
    make_trie(*t, *bump, begin, end, index, domain);

}

)xxx";

int main(int argc, char* argv[])
{
  size_t numBlocks  = 32;
  size_t numThreads = 128;
  // Create an instance of nvrtcProgram with the code string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(
    &prog, // prog
    trie, // buffer
    "trie.cu", // name
    0, // numHeaders
    nullptr, // headers
    nullptr)); // includeNames

  const char* opts[] = {
    "-std=c++11",
    "-I/usr/local/cuda/include",
    "-I../../include",
    "--gpu-architecture=compute_70",
    "--relocatable-device-code=true",
    "-default-device"};
  nvrtcResult compileResult = nvrtcCompileProgram(
    prog, // prog
    6, // numOptions
    opts); // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char* log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  return 0;
}
