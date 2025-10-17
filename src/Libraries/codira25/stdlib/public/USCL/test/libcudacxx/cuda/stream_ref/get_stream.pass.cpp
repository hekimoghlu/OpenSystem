/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 24, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/type_traits>
#include <uscl/stream_ref>

__host__ __device__ void test()
{
  ::cudaStream_t stream = reinterpret_cast<::cudaStream_t>(42);
  { // Can call get_stream on a cudaStream_t
    auto ref = ::cuda::get_stream(stream);
    assert(stream == ref);
  }

  { // Can call get_stream on a type with a get_stream method
    struct with_const_get_stream
    {
      ::cudaStream_t stream_;

      __host__ __device__ ::cuda::stream_ref get_stream() const noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    with_const_get_stream str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // Cannot call get_stream on a type with a non-const get_stream method
    struct with_mutable_get_stream
    {
      ::cudaStream_t stream_;

      __host__ __device__ ::cuda::stream_ref get_stream() noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const with_mutable_get_stream&>);
  }

  { // The get_stream method can return something convertible to cuda::stream_ref
    struct returns_convertible_to_stream_ref
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t get_stream() const noexcept
      {
        return stream_;
      }
    };
    returns_convertible_to_stream_ref str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // Cannot call get_stream on a type with a non-const get_stream method
    struct returns_not_convertible_to_stream_ref
    {
      __host__ __device__ int get_stream() const noexcept
      {
        return 42;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const returns_not_convertible_to_stream_ref&>);
  }

  { // The get_stream method works with queries
    struct with_query
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cuda::stream_ref query(::cuda::get_stream_t) const noexcept
      {
        return ::cuda::stream_ref{stream_};
      }
    };
    with_query str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // The get_stream method works with queries that return something convertible to stream_ref
    struct with_query_convertible_to_stream_ref
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t query(::cuda::get_stream_t) const noexcept
      {
        return stream_;
      }
    };
    with_query_convertible_to_stream_ref str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // The get_stream method works with types that have a stream member function
    struct with_stream
    {
      ::cudaStream_t stream_{};

      __host__ __device__ ::cudaStream_t stream() const noexcept
      {
        return stream_;
      }
    };
    with_stream str{stream};
    auto ref = ::cuda::get_stream(str);
    assert(stream == ref);
  }

  { // Cannot call get_stream on a type with a non-const get_stream method
    struct with_query_not_convertible_to_stream_ref
    {
      __host__ __device__ int query(::cuda::get_stream_t) const noexcept
      {
        return 42;
      }
    };
    static_assert(!::cuda::std::is_invocable_v<::cuda::get_stream_t, const with_query_not_convertible_to_stream_ref&>);
  }
}

int main(int argc, char** argv)
{
  test();

  return 0;
}
