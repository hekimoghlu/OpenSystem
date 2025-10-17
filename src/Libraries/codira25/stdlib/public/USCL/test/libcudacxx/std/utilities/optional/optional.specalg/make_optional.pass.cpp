/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
//
// template <class T>
//   constexpr optional<decay_t<T>> make_optional(T&& v);

#include <uscl/std/optional>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif
#ifdef _LIBCUDACXX_HAS_MEMORY
#  include <cuda/std/memory>
#else
#  include "MoveOnly.h"
#endif
#include <uscl/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
  {
    int arr[10];
    auto opt = cuda::std::make_optional(arr);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<int*>>);
    assert(*opt == arr);
  }

  {
    constexpr auto opt = cuda::std::make_optional(2);
    static_assert(cuda::std::is_same_v<decltype(opt), const cuda::std::optional<int>>);
    static_assert(opt.value() == 2, "");
  }

  {
    auto opt = cuda::std::make_optional(2);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<int>>);
    assert(*opt == 2);
  }
#ifdef _LIBCUDACXX_HAS_STRING
  {
    const cuda::std::string s = "123";
    auto opt                  = cuda::std::make_optional(s);
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<cuda::std::string>>);
    assert(*opt == "123");
  }
#endif
#ifdef _LIBCUDACXX_HAS_MEMORY
  {
    cuda::std::unique_ptr<int> s = cuda::std::make_unique<int>(3);
    auto opt                     = cuda::std::make_optional(cuda::std::move(s));
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<cuda::std::unique_ptr<int>>>);
    assert(**opt == 3);
    assert(s == nullptr);
  }
#else
  {
    MoveOnly s = 3;
    auto opt   = cuda::std::make_optional(cuda::std::move(s));
    static_assert(cuda::std::is_same_v<decltype(opt), cuda::std::optional<MoveOnly>>);
    assert(*opt == 3);
    assert(s == 0);
  }
#endif

  return 0;
}
