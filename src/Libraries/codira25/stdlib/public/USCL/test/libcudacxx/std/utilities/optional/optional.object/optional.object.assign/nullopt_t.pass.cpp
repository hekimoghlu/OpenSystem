/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// optional<T>& operator=(nullopt_t) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::nullopt;
using cuda::std::nullopt_t;
using cuda::std::optional;

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  enum class State
  {
    inactive,
    constructed,
    destroyed
  };
  State state = State::inactive;

  struct StateTracker
  {
    __host__ __device__ TEST_CONSTEXPR_CXX20 StateTracker(State& s)
        : state_(&s)
    {
      s = State::constructed;
    }
    __host__ __device__ TEST_CONSTEXPR_CXX20 ~StateTracker()
    {
      *state_ = State::destroyed;
    }

    State* state_;
  };
  {
    optional<int> opt{};
    static_assert(noexcept(opt = nullopt) == true, "");
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<int> opt(3);
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
#ifdef CCCL_ENABLE_OPTIONAL_REF
  {
    optional<int&> opt{};
    static_assert(noexcept(opt = nullopt) == true, "");
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    int val{3};
    optional<int&> opt(val);
    opt = nullopt;
    assert(static_cast<bool>(opt) == false);
  }
#endif // CCCL_ENABLE_OPTIONAL_REF
  {
    optional<StateTracker> opt{};
    opt = nullopt;
    assert(state == State::inactive);
    assert(static_cast<bool>(opt) == false);
  }
  {
    optional<StateTracker> opt(state);
    assert(state == State::constructed);
    opt = nullopt;
    assert(state == State::destroyed);
    assert(static_cast<bool>(opt) == false);
  }
  return true;
}

int main(int, char**)
{
#if TEST_STD_VER > 2017
  static_assert(test());
#endif
  test();
  using TT = TestTypes::TestType;
  TT::reset();
  {
    optional<TT> opt{};
    static_assert(noexcept(opt = nullopt) == true, "");
    assert(TT::destroyed() == 0);
    opt = nullopt;
    assert(TT::constructed() == 0);
    assert(TT::alive() == 0);
    assert(TT::destroyed() == 0);
    assert(static_cast<bool>(opt) == false);
  }
  assert(TT::alive() == 0);
  assert(TT::destroyed() == 0);
  TT::reset();
  {
    optional<TT> opt(42);
    assert(TT::destroyed() == 0);
    TT::reset_constructors();
    opt = nullopt;
    assert(TT::constructed() == 0);
    assert(TT::alive() == 0);
    assert(TT::destroyed() == 1);
    assert(static_cast<bool>(opt) == false);
  }
  assert(TT::alive() == 0);
  assert(TT::destroyed() == 1);
  TT::reset();

  return 0;
}
