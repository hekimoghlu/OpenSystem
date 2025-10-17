/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 26, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

#include <uscl/std/cassert>
#include <uscl/std/optional>

#include "helpers.h"

template <int Value>
struct int_generator
{
  __host__ __device__ auto operator()() const
  {
    return Value;
  }
};

struct nullopt_generator
{
  __host__ __device__ auto operator()() const
  {
    return cuda::std::nullopt;
  }
};

template <typename OperandGenerator>
struct assign_tester
{
  template <typename Opt>
  __host__ __device__ static void initialize(Opt& opt)
  {
    opt = OperandGenerator()();
  }

  template <typename Opt>
  __host__ __device__ static void validate(Opt& opt)
  {
    assert(opt == OperandGenerator()());
  }
};

struct reset_tester
{
  template <typename Opt>
  __host__ __device__ static void initialize(Opt& opt)
  {
    opt.reset();
  }

  template <typename Opt>
  __host__ __device__ static void validate(Opt& opt)
  {
    assert(!opt.has_value());
  }
};

template <typename OperandGenerator>
struct swap_tester
{
  template <typename Opt>
  __host__ __device__ static void initialize(Opt& opt)
  {
    auto original                                     = opt;
    cuda::std::optional<typename Opt::value_type> val = OperandGenerator()();
    opt.swap(val);
    assert(val == original);
  }

  template <typename Opt>
  __host__ __device__ static void validate(Opt& opt)
  {
    cuda::std::optional<typename Opt::value_type> val = OperandGenerator()();
    assert(opt == val);
  }
};

using testers =
  tester_list<assign_tester<nullopt_generator>,
              assign_tester<int_generator<123>>,
              assign_tester<nullopt_generator>,
              swap_tester<int_generator<17>>,
              reset_tester,
              assign_tester<int_generator<31>>,
              swap_tester<nullopt_generator>>;

struct non_trivial
{
  int i;

  __host__ __device__ non_trivial(int i)
      : i(i)
  {}
  non_trivial(const non_trivial&)            = default;
  non_trivial& operator=(const non_trivial&) = default;

  __host__ __device__ friend bool operator==(non_trivial lhs, non_trivial rhs)
  {
    return lhs.i == rhs.i;
  }
};

void kernel_invoker()
{
  // TODO: add validate_movable
  validate_pinned<cuda::std::optional<int>, testers>();
  validate_pinned<cuda::std::optional<non_trivial>, testers>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (kernel_invoker();))

  return 0;
}
