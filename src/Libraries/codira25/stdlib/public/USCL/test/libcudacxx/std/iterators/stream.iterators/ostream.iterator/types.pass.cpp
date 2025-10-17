/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 14, 2021.
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

// template <class T, class charT = char, class traits = char_traits<charT>,
//           class Distance = ptrdiff_t>
// class ostream_iterator
//     : public iterator<output_iterator_tag, void, void, void, void>
// {
// public:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef basic_istream<charT,traits> istream_type;
//     ...

#include <uscl/std/iterator>
#include <uscl/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::ostream_iterator<double> I1;

  static_assert((cuda::std::is_same<I1::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I1::value_type, void>::value), "");
#if TEST_STD_VER > 2017
  static_assert((cuda::std::is_same<I1::difference_type, ptrdiff_t>::value), "");
#else
  static_assert((cuda::std::is_same<I1::difference_type, void>::value), "");
#endif
  static_assert((cuda::std::is_same<I1::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I1::reference, void>::value), "");
  static_assert((cuda::std::is_same<I1::char_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::traits_type, cuda::std::char_traits<char>>::value), "");
  static_assert((cuda::std::is_same<I1::ostream_type, cuda::std::ostream>::value), "");
  typedef cuda::std::ostream_iterator<unsigned, wchar_t> I2;

  static_assert((cuda::std::is_same<I2::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I2::value_type, void>::value), "");
#if TEST_STD_VER > 2017
  static_assert((cuda::std::is_same<I2::difference_type, ptrdiff_t>::value), "");
#else
  static_assert((cuda::std::is_same<I2::difference_type, void>::value), "");
#endif
  static_assert((cuda::std::is_same<I2::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I2::reference, void>::value), "");
  static_assert((cuda::std::is_same<I2::char_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::traits_type, cuda::std::char_traits<wchar_t>>::value), "");
  static_assert((cuda::std::is_same<I2::ostream_type, cuda::std::wostream>::value), "");

  return 0;
}
