/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

// UNSUPPORTED: nvrtc
// UNSUPPORTED: msvc-19.16

// cuda::std::ranges::rend

#include <uscl/std/ranges>

struct NonBorrowedRange
{
  __host__ __device__ int* begin() const;
  __host__ __device__ int* end() const;
};
static_assert(!cuda::std::ranges::enable_borrowed_range<NonBorrowedRange>);

// Verify that if the expression is an rvalue and `enable_borrowed_range` is false, `ranges::rend` is ill-formed.
__host__ __device__ void test()
{
  cuda::std::ranges::rend(NonBorrowedRange());
  // expected-error-re@-1 {{{{call to deleted function call operator in type 'const
  // (cuda::std::ranges::)?__rend::__fn'}}}} expected-error@-2  {{attempt to use a deleted function}}
}

int main(int, char**)
{
  return 0;
}
