/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 20, 2022.
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

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <size_t I, class... Types>
//   const typename tuple_element<I, tuple<Types...> >::type&&
//   get(const tuple<Types...>&& t);

#include <uscl/std/tuple>
#include <uscl/std/utility>
// cuda::std::string not supported
// #include <uscl/std/string>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  {
    using T = cuda::std::tuple<int>;
    const T t(3);
    static_assert(cuda::std::is_same<const int&&, decltype(cuda::std::get<0>(cuda::std::move(t)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(t))), "");
    const int&& i = cuda::std::get<0>(cuda::std::move(t));
    assert(i == 3);
  }

  // cuda::std::string not supported
  /*
  {
  using T = cuda::std::tuple<cuda::std::string, int>;
  const T t("high", 5);
  static_assert(cuda::std::is_same<const cuda::std::string&&, decltype(cuda::std::get<0>(cuda::std::move(t)))>::value,
  ""); static_assert(noexcept(cuda::std::get<0>(cuda::std::move(t))), ""); static_assert(cuda::std::is_same<const int&&,
  decltype(cuda::std::get<1>(cuda::std::move(t)))>::value, "");
  static_assert(noexcept(cuda::std::get<1>(cuda::std::move(t))), "");
  const cuda::std::string&& s = cuda::std::get<0>(cuda::std::move(t));
  const int&& i = cuda::std::get<1>(cuda::std::move(t));
  assert(s == "high");
  assert(i == 5);
  }
  */

  {
    int x       = 42;
    int const y = 43;
    cuda::std::tuple<int&, int const&> const p(x, y);
    static_assert(cuda::std::is_same<int&, decltype(cuda::std::get<0>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&, decltype(cuda::std::get<1>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(p))), "");
  }

  {
    int x       = 42;
    int const y = 43;
    cuda::std::tuple<int&&, int const&&> const p(cuda::std::move(x), cuda::std::move(y));
    static_assert(cuda::std::is_same<int&&, decltype(cuda::std::get<0>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<0>(cuda::std::move(p))), "");
    static_assert(cuda::std::is_same<int const&&, decltype(cuda::std::get<1>(cuda::std::move(p)))>::value, "");
    static_assert(noexcept(cuda::std::get<1>(cuda::std::move(p))), "");
  }

  {
    using T = cuda::std::tuple<double, int>;
    constexpr const T t(2.718, 5);
    static_assert(cuda::std::get<0>(cuda::std::move(t)) == 2.718, "");
    static_assert(cuda::std::get<1>(cuda::std::move(t)) == 5, "");
  }

  return 0;
}
