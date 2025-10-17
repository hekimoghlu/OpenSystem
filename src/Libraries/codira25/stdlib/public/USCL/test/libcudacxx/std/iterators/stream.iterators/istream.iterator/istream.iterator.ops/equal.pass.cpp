/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 3, 2022.
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

// <cuda/std/iterator>

// class istream_iterator

// template <class T, class charT, class traits, class Distance>
//   bool operator==(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);
//
// template <class T, class charT, class traits, class Distance>
//   bool operator!=(const istream_iterator<T,charT,traits,Distance> &x,
//                   const istream_iterator<T,charT,traits,Distance> &y);

#include <uscl/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  cuda::std::istringstream inf1(" 1 23");
  cuda::std::istringstream inf2(" 1 23");
  cuda::std::istream_iterator<int> i1(inf1);
  cuda::std::istream_iterator<int> i2(inf1);
  cuda::std::istream_iterator<int> i3(inf2);
  cuda::std::istream_iterator<int> i4;
  cuda::std::istream_iterator<int> i5;
  assert(i1 == i1);
  assert(i1 == i2);
  assert(i1 != i3);
  assert(i1 != i4);
  assert(i1 != i5);

  assert(i2 == i2);
  assert(i2 != i3);
  assert(i2 != i4);
  assert(i2 != i5);

  assert(i3 == i3);
  assert(i3 != i4);
  assert(i3 != i5);

  assert(i4 == i4);
  assert(i4 == i5);

  assert(cuda::std::operator==(i1, i2));
  assert(cuda::std::operator!=(i1, i3));

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
