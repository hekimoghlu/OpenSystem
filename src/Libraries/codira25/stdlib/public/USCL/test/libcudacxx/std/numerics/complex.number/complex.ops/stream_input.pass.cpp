/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// UNSUPPORTED: nvrtc

// <complex>

// template<class T, class charT, class traits>
//   basic_istream<charT, traits>&
//   operator>>(basic_istream<charT, traits>& is, complex<T>& x);

#include <uscl/std/cassert>
#include <uscl/std/complex>

#include <sstream>

#include "test_macros.h"

template <class T>
void test()
{
  {
    std::istringstream is("5");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.eof());
  }
  {
    std::istringstream is(" 5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" , 5, ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is(" ( 5.5 ) ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("  5.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(5.5, 0));
    assert(is.good());
  }
  {
    std::istringstream is("(5.5 ");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("(5.5,");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(0, 0));
    assert(is.fail());
  }
  {
    std::istringstream is("( -5.5 , -6.5 )");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
  {
    std::istringstream is("(-5.5,-6.5)");
    cuda::std::complex<T> c;
    is >> c;
    assert(c == cuda::std::complex<T>(-5.5, -6.5));
    assert(!is.eof());
  }
}

void test()
{
  test<float>();
  test<double>();
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test();)
  return 0;
}
