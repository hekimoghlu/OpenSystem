/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 20, 2025.
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
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// template <size_t I, class T, size_t N> T& get(array<T, N>& a);

// Prevent -Warray-bounds from issuing a diagnostic when testing with clang verify.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -Wno-array-bounds

#include <uscl/std/array>
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c                  = {1, 2, 3.5};
    cuda::std::get<3>(c) = 5.5; // expected-note {{requested here}}
    // expected-error-re@array:* {{{{(static_assert|static assertion)}} failed{{( due to requirement '3U[L]{0,2} <
    // 3U[L]{0,2}')?}}{{.*}}Index out of bounds in cuda::std::get<> (cuda::std::array)}}
  }

  return 0;
}
