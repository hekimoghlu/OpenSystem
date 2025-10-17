/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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

// UNSUPPORTED: LIBCUDACXX-has-no-incomplete-ranges

// Test the [[nodiscard]] extension in libc++.

// template<class I>
// unspecified iter_move;

#include <uscl/std/iterator>

struct WithADL
{
  WithADL() = default;
  __host__ __device__ constexpr decltype(auto) operator*() const noexcept;
  __host__ __device__ constexpr WithADL& operator++() noexcept;
  __host__ __device__ constexpr void operator++(int) noexcept;
  __host__ __device__ constexpr bool operator==(WithADL const&) const noexcept;
  __host__ __device__ friend constexpr auto iter_move(WithADL&)
  {
    return 0;
  }
};

int main(int, char**)
{
  int* noADL = nullptr;
  cuda::std::ranges::iter_move(noADL); // expected-warning {{ignoring return value of function declared with 'nodiscard'
                                       // attribute}}

  WithADL adl;
  cuda::std::ranges::iter_move(adl); // expected-warning {{ignoring return value of function declared with 'nodiscard'
                                     // attribute}}

  return 0;
}
