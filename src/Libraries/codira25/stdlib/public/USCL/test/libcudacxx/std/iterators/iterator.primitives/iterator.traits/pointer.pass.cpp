/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

// template<class T>
// struct iterator_traits<T*>
// {
//   typedef ptrdiff_t                  difference_type;
//   typedef T                          value_type;
//   typedef T*                         pointer;
//   typedef T&                         reference;
//   typedef random_access_iterator_tag iterator_category;
// };

#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct A
{};

int main(int, char**)
{
  typedef cuda::std::iterator_traits<A*> It;
  static_assert((cuda::std::is_same<It::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<It::value_type, A>::value), "");
  static_assert((cuda::std::is_same<It::pointer, A*>::value), "");
  static_assert((cuda::std::is_same<It::reference, A&>::value), "");
  static_assert((cuda::std::is_same<It::iterator_category, cuda::std::random_access_iterator_tag>::value), "");

  return 0;
}
