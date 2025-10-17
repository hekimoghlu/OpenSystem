/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 27, 2025.
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

//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/stream_ref>

static_assert(cuda::std::is_default_constructible<cuda::stream_ref>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, int>::value, "");
static_assert(!cuda::std::is_constructible<cuda::stream_ref, cuda::std::nullptr_t>::value, "");

template <class T, class = void>
constexpr bool has_value_type = false;

template <class T>
constexpr bool has_value_type<T, cuda::std::void_t<typename T::value_type>> = true;
static_assert(has_value_type<cuda::stream_ref>, "");

__host__ __device__ void test()
{
  { // from stream
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(42);
    cuda::stream_ref ref{stream};
    static_assert(noexcept(cuda::stream_ref{stream}), "");
    assert(ref.get() == reinterpret_cast<cudaStream_t>(42));
  }
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
