/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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

// <functional>

// template <class T>
//   reference_wrapper(T&) -> reference_wrapper<T>;

// #include <uscl/std/functional>
#include <uscl/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  int i = 0;
  cuda::std::reference_wrapper ri(i);
  static_assert(cuda::std::is_same_v<decltype(ri), cuda::std::reference_wrapper<int>>);
  cuda::std::reference_wrapper ri2(ri);
  static_assert(cuda::std::is_same_v<decltype(ri2), cuda::std::reference_wrapper<int>>);
  unused(ri2);

  const int j = 0;
  cuda::std::reference_wrapper rj(j);
  static_assert(cuda::std::is_same_v<decltype(rj), cuda::std::reference_wrapper<const int>>);
  cuda::std::reference_wrapper rj2(rj);
  static_assert(cuda::std::is_same_v<decltype(rj2), cuda::std::reference_wrapper<const int>>);
  unused(rj2);

  return 0;
}
