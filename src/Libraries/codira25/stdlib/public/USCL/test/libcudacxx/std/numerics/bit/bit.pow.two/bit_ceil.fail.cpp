/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// template <class T>
//   constexpr T bit_ceil(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <uscl/std/bit>
#include <uscl/std/cassert>
#include <uscl/std/cstdint>
#include <uscl/std/limits>

#include "test_macros.h"

class A
{};
enum E1 : unsigned char
{
  rEd
};
enum class E2 : unsigned char
{
  red
};

template <typename T>
__host__ __device__ constexpr bool toobig()
{
  return 0 == cuda::std::bit_ceil(cuda::std::numeric_limits<T>::max());
}

int main(int, char**)
{
  //	Make sure we generate a compile-time error for UB
  static_assert(toobig<unsigned char>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned short>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                               // not an integral constant expression}}
  static_assert(toobig<unsigned>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<unsigned long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
  static_assert(toobig<unsigned long long>(), ""); // expected-error-re {{{{(static_assert|static assertion)}}
                                                   // expression is not an integral constant expression}}

  static_assert(toobig<uint8_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                        // integral constant expression}}
  static_assert(toobig<uint16_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint32_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<uint64_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                         // an integral constant expression}}
  static_assert(toobig<size_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                       // integral constant expression}}
  static_assert(toobig<uintmax_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}
  static_assert(toobig<uintptr_t>(), ""); // expected-error-re {{{{(static_assert|static assertion)}} expression is not
                                          // an integral constant expression}}

  return 0;
}
