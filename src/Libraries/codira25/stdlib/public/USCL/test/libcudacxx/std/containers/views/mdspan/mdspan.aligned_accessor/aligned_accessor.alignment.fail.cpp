/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <uscl/std/mdspan>

#include <test_macros.h>

int main(int, char**)
{
  using T = int;
  // conversion from smaller aligned accessor
  cuda::std::aligned_accessor<T, sizeof(T)> aligned_x1{};
  cuda::std::aligned_accessor<T, sizeof(T) * 2> aligned_x2{aligned_x1};
  unused(aligned_x2);

  // alignment to small
  cuda::std::aligned_accessor<T, sizeof(T) / 2> aligned_half{};
  unused(aligned_half);

  // alignment non-power of 2
  cuda::std::aligned_accessor<T, 6> aligned6{};
  unused(aligned_half);

  // non-convertible
  cuda::std::aligned_accessor<T, sizeof(T)> aligned_int{};
  cuda::std::aligned_accessor<float, sizeof(T)> aligned_float{aligned_int};
  unused(aligned_float);
  return 0;
}
