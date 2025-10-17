/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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

// We cannot suppress execution checks in cuda::std::construct_at
// UNSUPPORTED: clang-14

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "host_device_types.h"
#include "test_macros.h"

template <class T>
__device__ void test()
{
  using optional = cuda::std::optional<T>;
  { // default construction
    optional default_constructed{};
    assert(!default_constructed.has_value());
  }

  if constexpr (!cuda::std::is_reference_v<T>)
  { // in_place zero initialization
    optional in_place_zero_initialization{cuda::std::in_place};
    assert(in_place_zero_initialization.has_value());
    assert(*in_place_zero_initialization == 0);
  }

  cuda::std::remove_reference_t<T> val{42};
  { // in_place initialization
    optional in_place_initialization{cuda::std::in_place, val};
    assert(in_place_initialization.has_value());
    assert(*in_place_initialization == 42);
  }

  { // value initialization
    optional value_initialization{val};
    assert(value_initialization.has_value());
    assert(*value_initialization == 42);
  }

  { // copy construction
    optional input{val};
    optional dest{input};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // move construction
    optional input{val};
    optional dest{cuda::std::move(input)};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  cuda::std::remove_reference_t<T> other_val{1337};
  { // assignment, value to value
    optional input{val};
    optional dest{other_val};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to empty
    optional input{val};
    optional dest{};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, empty to value
    optional input{};
    optional dest{other_val};
    dest = input;
    assert(!dest.has_value());
  }

  { // assignment, empty to empty
    optional input{};
    optional dest{};
    dest = input;
    assert(!dest.has_value());
  }

  { // comparison with optional
    optional lhs{val};
    optional rhs{other_val};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // comparison with type
    optional opt{val};
    assert(opt == device_only_type{val});
    assert(device_only_type{val} == opt);
    assert(opt != device_only_type{other_val});
    assert(device_only_type{other_val} != opt);

    assert(opt < device_only_type{other_val});
    assert(device_only_type{7} < opt);
    assert(opt <= device_only_type{other_val});
    assert(device_only_type{7} <= opt);

    assert(opt > device_only_type{7});
    assert(device_only_type{other_val} > opt);
    assert(opt >= device_only_type{7});
    assert(device_only_type{other_val} >= opt);
  }

  { // swap
    optional lhs{val};
    optional rhs{other_val};
    lhs.swap(rhs);
    assert(*lhs == 1337);
    assert(*rhs == 42);

    swap(lhs, rhs);
    assert(*lhs == 42);
    assert(*rhs == 1337);
  }
}

__device__ void test()
{
  test<device_only_type>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<device_only_type&>();
#endif // CCCL_ENABLE_OPTIONAL_REF
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
