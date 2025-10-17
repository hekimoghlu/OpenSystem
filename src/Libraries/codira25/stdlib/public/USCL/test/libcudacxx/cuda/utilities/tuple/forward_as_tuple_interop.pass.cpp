/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

// UNSUPPORTED: nvrtc
#include <uscl/std/cassert>
#include <uscl/std/tuple>
#include <uscl/std/type_traits>

#include <nv/target>

#include <tuple>

constexpr bool test()
{
  // Ensure we can use std:: types inside cuda::std::make_tuple
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::make_tuple(cuda::std::integral_constant<int, 42>(), std::integral_constant<int, 1337>());
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  // Ensure we can use std:: types inside cuda::std::tuple_cat
  {
    using ret = cuda::std::tuple<cuda::std::integral_constant<int, 42>, std::integral_constant<int, 1337>>;
    auto t    = cuda::std::tuple_cat(cuda::std::make_tuple(cuda::std::integral_constant<int, 42>()),
                                  cuda::std::make_tuple(std::integral_constant<int, 1337>()));
    static_assert(cuda::std::is_same<decltype(t), ret>::value, "");
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
  }

  return true;
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test(); static_assert(test(), "");));

  return 0;
}
