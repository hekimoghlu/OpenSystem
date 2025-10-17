/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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

// class ostreambuf_iterator

// bool failed() const throw();

#include <uscl/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

template <typename Char, typename Traits = cuda::std::char_traits<Char>>
struct my_streambuf : public cuda::std::basic_streambuf<Char, Traits>
{
  typedef typename cuda::std::basic_streambuf<Char, Traits>::int_type int_type;
  typedef typename cuda::std::basic_streambuf<Char, Traits>::char_type char_type;

  my_streambuf() {}
  int_type sputc(char_type)
  {
    return Traits::eof();
  }
};

int main(int, char**)
{
  {
    my_streambuf<char> buf;
    cuda::std::ostreambuf_iterator<char> i(&buf);
    i = 'a';
    assert(i.failed());
  }
  {
    my_streambuf<wchar_t> buf;
    cuda::std::ostreambuf_iterator<wchar_t> i(&buf);
    i = L'a';
    assert(i.failed());
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
