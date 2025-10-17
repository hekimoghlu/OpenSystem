/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 16, 2023.
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

// template <class _Iterator>
// struct __bounded_iter;
//
// Nested types

#include <uscl/std/cstddef>
#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

#if TEST_STD_VER > 2017
struct Iterator
{
  struct value_type
  {};
  using difference_type = int;
  struct pointer
  {};
  using reference = value_type&;
  struct iterator_category : cuda::std::random_access_iterator_tag
  {};
  using iterator_concept = cuda::std::contiguous_iterator_tag;
};

using BoundedIter1 = cuda::std::__bounded_iter<Iterator>;
static_assert(cuda::std::is_same<BoundedIter1::value_type, Iterator::value_type>::value, "");
static_assert(cuda::std::is_same<BoundedIter1::difference_type, Iterator::difference_type>::value, "");
static_assert(cuda::std::is_same<BoundedIter1::pointer, Iterator::pointer>::value, "");
static_assert(cuda::std::is_same<BoundedIter1::reference, Iterator::reference>::value, "");
static_assert(cuda::std::is_same<BoundedIter1::iterator_category, Iterator::iterator_category>::value, "");
static_assert(cuda::std::is_same<BoundedIter1::iterator_concept, Iterator::iterator_concept>::value, "");
#endif // TEST_STD_VER > 2017

using BoundedIter2 = cuda::std::__bounded_iter<int*>;
static_assert(cuda::std::is_same<BoundedIter2::value_type, int>::value, "");
static_assert(cuda::std::is_same<BoundedIter2::difference_type, cuda::std::ptrdiff_t>::value, "");
static_assert(cuda::std::is_same<BoundedIter2::pointer, int*>::value, "");
static_assert(cuda::std::is_same<BoundedIter2::reference, int&>::value, "");
static_assert(cuda::std::is_same<BoundedIter2::iterator_category, cuda::std::random_access_iterator_tag>::value, "");
#if TEST_STD_VER > 2017
static_assert(cuda::std::is_same<BoundedIter2::iterator_concept, cuda::std::contiguous_iterator_tag>::value, "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
