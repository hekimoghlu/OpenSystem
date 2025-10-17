/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 2, 2025.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types>
// constexpr bool
// operator==(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator!=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator<=(variant<Types...> const&, variant<Types...> const&) noexcept;
//
// template <class ...Types>
// constexpr bool
// operator>=(variant<Types...> const&, variant<Types...> const&) noexcept;

#include <uscl/std/cassert>
#include <uscl/std/type_traits>
#include <uscl/std/utility>
#include <uscl/std/variant>

#include "test_macros.h"

struct MyBoolExplicit
{
  bool value;
  constexpr explicit MyBoolExplicit(bool v)
      : value(v)
  {}
  constexpr explicit operator bool() const noexcept
  {
    return value;
  }
};

struct ComparesToMyBoolExplicit
{
  int value = 0;
};
inline constexpr MyBoolExplicit
operator==(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value == RHS.value);
}
inline constexpr MyBoolExplicit
operator!=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value != RHS.value);
}
inline constexpr MyBoolExplicit
operator<(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value < RHS.value);
}
inline constexpr MyBoolExplicit
operator<=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value <= RHS.value);
}
inline constexpr MyBoolExplicit
operator>(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value > RHS.value);
}
inline constexpr MyBoolExplicit
operator>=(const ComparesToMyBoolExplicit& LHS, const ComparesToMyBoolExplicit& RHS) noexcept
{
  return MyBoolExplicit(LHS.value >= RHS.value);
}

int main(int, char**)
{
  using V = cuda::std::variant<int, ComparesToMyBoolExplicit>;
  V v1(42);
  V v2(101);
  // expected-error-re@variant:* 6 {{{{(static_assert|static assertion)}} failed{{.*}}the relational operator does not
  // return a type which is implicitly convertible to bool}} expected-error@variant:* 6 {{no viable conversion}}
  (void) (v1 == v2); // expected-note {{here}}
  (void) (v1 != v2); // expected-note {{here}}
  (void) (v1 < v2); // expected-note {{here}}
  (void) (v1 <= v2); // expected-note {{here}}
  (void) (v1 > v2); // expected-note {{here}}
  (void) (v1 >= v2); // expected-note {{here}}

  return 0;
}
