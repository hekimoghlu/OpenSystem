/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 26, 2023.
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

// <utility>

// template<class T, T... I>
// struct integer_sequence
// {
//     typedef T type;
//
//     static constexpr size_t size() noexcept;
// };

#include <uscl/std/cassert>
#include <uscl/std/cstddef>
#include <uscl/std/type_traits>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  //  Make a few of sequences
  using intseq3    = cuda::std::integer_sequence<int, 3, 2, 1>;
  using size1      = cuda::std::integer_sequence<cuda::std::size_t, 7>;
  using ushortseq2 = cuda::std::integer_sequence<unsigned short, 4, 6>;
  using bool0      = cuda::std::integer_sequence<bool>;

  //  Make sure they're what we expect
  static_assert(cuda::std::is_same<intseq3::value_type, int>::value, "intseq3 type wrong");
  static_assert(intseq3::size() == 3, "intseq3 size wrong");

  static_assert(cuda::std::is_same<size1::value_type, cuda::std::size_t>::value, "size1 type wrong");
  static_assert(size1::size() == 1, "size1 size wrong");

  static_assert(cuda::std::is_same<ushortseq2::value_type, unsigned short>::value, "ushortseq2 type wrong");
  static_assert(ushortseq2::size() == 2, "ushortseq2 size wrong");

  static_assert(cuda::std::is_same<bool0::value_type, bool>::value, "bool0 type wrong");
  static_assert(bool0::size() == 0, "bool0 size wrong");

  return 0;
}
