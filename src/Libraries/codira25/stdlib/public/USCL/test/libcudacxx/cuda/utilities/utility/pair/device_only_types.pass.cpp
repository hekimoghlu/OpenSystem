/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 5, 2022.
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
#include <uscl/std/utility>

#include "host_device_types.h"
#include "test_macros.h"

__device__ void test()
{
  using pair = cuda::std::pair<device_only_type, device_only_type>;
  { // default construction
    pair default_constructed{};
    assert(default_constructed.first == 0);
    assert(default_constructed.second == 0);
  }

  { // value initialization
    pair value_initialization{device_only_type{42}, device_only_type{1337}};
    assert(value_initialization.first == 42);
    assert(value_initialization.second == 1337);
  }

  { // value initialization
    pair value_initialization{42, 1337};
    assert(value_initialization.first == 42);
    assert(value_initialization.second == 1337);
  }

  { // copy construction
    pair input{42, 1337};
    pair dest{input};
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // move construction
    pair input{42, 1337};
    pair dest{cuda::std::move(input)};
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // assignment, value to value
    pair input{42, 1337};
    pair dest{1337, 42};
    dest = input;
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // comparison with pair
    pair lhs{42, 1337};
    pair rhs{1337, 42};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // swap
    pair lhs{42, 1337};
    pair rhs{1337, 42};
    lhs.swap(rhs);
    assert(lhs.first == 1337);
    assert(lhs.second == 42);
    assert(rhs.first == 42);
    assert(rhs.second == 1337);

    swap(lhs, rhs);
    assert(lhs.first == 42);
    assert(lhs.second == 1337);
    assert(rhs.first == 1337);
    assert(rhs.second == 42);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
