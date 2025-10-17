/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 16, 2023.
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
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class E2> friend constexpr bool operator==(const expected& x, const unexpected<E2>& e);

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/expected>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

struct Data
{
  int i;
  __host__ __device__ constexpr Data(int ii)
      : i(ii)
  {}

  __host__ __device__ friend constexpr bool operator==(const Data& data, int ii)
  {
    return data.i == ii;
  }
#if TEST_STD_VER < 2020
  __host__ __device__ friend constexpr bool operator!=(const Data& data, int ii)
  {
    return data.i != ii;
  }
#endif // TEST_STD_VER < 2020
};

__host__ __device__ constexpr bool test()
{
  // x.has_value()
  {
    const cuda::std::expected<void, Data> e1;
    cuda::std::unexpected<int> un2(10);
    cuda::std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 != un3);
  }

  // !x.has_value()
  {
    const cuda::std::expected<void, Data> e1(cuda::std::unexpect, 5);
    cuda::std::unexpected<int> un2(10);
    cuda::std::unexpected<int> un3(5);
    assert(e1 != un2);
    assert(e1 == un3);
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
