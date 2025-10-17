/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/stream_ref>

#include <atomic>
#include <chrono>
#include <thread>

#include "test_macros.h"

void CUDART_CB callback(cudaStream_t, cudaError_t, void* flag)
{
  std::chrono::milliseconds sleep_duration{1000};
  std::this_thread::sleep_for(sleep_duration);
  assert(!reinterpret_cast<std::atomic_flag*>(flag)->test_and_set());
}

void test_sync(cuda::stream_ref& ref)
{
#if TEST_HAS_EXCEPTIONS()
  try
  {
    ref.sync();
  }
  catch (...)
  {
    assert(false && "Should not have thrown");
  }
#else
  ref.sync();
#endif // TEST_HAS_EXCEPTIONS()
}

int main(int argc, char** argv)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    ( // passing case
      cudaStream_t stream; cudaStreamCreate(&stream); std::atomic_flag flag = ATOMIC_FLAG_INIT;
      cudaStreamAddCallback(stream, callback, &flag, 0);
      cuda::stream_ref ref{stream};
      test_sync(ref);
      assert(flag.test_and_set());
      cudaStreamDestroy(stream);))

  return 0;
}
