/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 10, 2023.
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
//
//===----------------------------------------------------------------------===//

// XFAIL: gcc-4

// <utility>

// template <class T1, class T2> struct pair

// explicit(see-below) constexpr pair();

// NOTE: The SFINAE on the default constructor is tested in
//       default-sfinae.pass.cpp

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "archetypes.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    typedef cuda::std::pair<float, short*> P;
    P p;
    assert(p.first == 0.0f);
    assert(p.second == nullptr);
  }
  {
    typedef cuda::std::pair<float, short*> P;
    constexpr P p;
    static_assert(p.first == 0.0f, "");
    static_assert(p.second == nullptr, "");
  }
  {
    using NoDefault = ImplicitTypes::NoDefault;
    using P         = cuda::std::pair<int, NoDefault>;
    static_assert(!cuda::std::is_default_constructible<P>::value, "");
    using P2 = cuda::std::pair<NoDefault, int>;
    static_assert(!cuda::std::is_default_constructible<P2>::value, "");
  }
  {
    struct Base
    {};
    struct Derived : Base
    {
    protected:
      Derived() = default;
    };
    static_assert(!cuda::std::is_default_constructible<cuda::std::pair<Derived, int>>::value, "");
  }

  return 0;
}
