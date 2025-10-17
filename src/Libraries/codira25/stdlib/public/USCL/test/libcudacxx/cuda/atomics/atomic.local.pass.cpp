/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: windows && pre-sm-70

#include <uscl/atomic>
#include <uscl/std/cassert>

template <typename T>
__device__ T store(T in)
{
  cuda::atomic<T> x(in);
  x.store(in + 1, cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T compare_exchange_weak(T in)
{
  cuda::atomic<T> x(in);
  T old = T(7);
  x.compare_exchange_weak(old, T(42), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T compare_exchange_strong(T in)
{
  cuda::atomic<T> x(in);
  T old = T(7);
  x.compare_exchange_strong(old, T(42), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T exchange(T in)
{
  cuda::atomic<T> x(in);
  T out = x.exchange(T(1), cuda::memory_order_relaxed);
  return out + x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_add(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_add(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_sub(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_sub(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_and(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_and(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_or(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_or(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_xor(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_xor(T(1), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_min(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_min(T(7), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ T fetch_max(T in)
{
  cuda::atomic<T> x(in);
  x.fetch_max(T(7), cuda::memory_order_relaxed);
  return x.load(cuda::memory_order_relaxed);
}

template <typename T>
__device__ inline void tests()
{
  const T tid = threadIdx.x;
  assert(tid + T(1) == store(tid));
  assert(T(1) + tid == exchange(tid));
  assert(tid == T(7) ? T(42) : tid == compare_exchange_weak(tid));
  assert(tid == T(7) ? T(42) : tid == compare_exchange_strong(tid));
  assert((tid + T(1)) == fetch_add(tid));
  assert((tid & T(1)) == fetch_and(tid));
  assert((tid | T(1)) == fetch_or(tid));
  assert((tid ^ T(1)) == fetch_xor(tid));
  assert(min(tid, T(7)) == fetch_min(tid));
  assert(max(tid, T(7)) == fetch_max(tid));
  assert(T(tid - T(1)) == fetch_sub(tid));
}

int main(int arg, char** argv)
{
#if !defined(_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)
  NV_IF_ELSE_TARGET(
    NV_IS_HOST,
    (cuda_thread_count = 64;),
    (tests<uint8_t>(); tests<uint16_t>(); tests<uint32_t>(); tests<uint64_t>(); tests<int8_t>(); tests<int16_t>();
     tests<int32_t>();
     tests<int64_t>();))
#endif
  return 0;
}
