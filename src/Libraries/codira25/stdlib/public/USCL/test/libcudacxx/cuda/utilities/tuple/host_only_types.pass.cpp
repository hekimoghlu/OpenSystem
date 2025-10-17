/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <uscl/std/cassert>
#include <uscl/std/tuple>

#include "host_device_types.h"
#include "test_macros.h"

void test()
{
  using tuple = cuda::std::tuple<host_only_type>;
  { // default construction
    tuple default_constructed{};
    assert(cuda::std::get<0>(default_constructed) == 0);
  }

  { // value initialization
    tuple value_initialization{host_only_type{42}};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // value initialization
    tuple value_initialization{42};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // copy construction
    tuple input{42};
    tuple dest{input};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // move construction
    tuple input{42};
    tuple dest{cuda::std::move(input)};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // assignment, value to value
    tuple input{42};
    tuple dest{1337};
    dest = input;
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // assignment, value to empty
    tuple input{42};
    tuple dest{};
    dest = input;
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // comparison with tuple
    tuple lhs{42};
    tuple rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // swap
    tuple lhs{42};
    tuple rhs{1337};
    lhs.swap(rhs);
    assert(cuda::std::get<0>(lhs) == 1337);
    assert(cuda::std::get<0>(rhs) == 42);

    swap(lhs, rhs);
    assert(cuda::std::get<0>(lhs) == 42);
    assert(cuda::std::get<0>(rhs) == 1337);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
