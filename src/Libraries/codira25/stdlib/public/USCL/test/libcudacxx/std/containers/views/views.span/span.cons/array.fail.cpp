/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 26, 2024.
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

// <span>

// template<size_t N>
//     constexpr span(element_type (&arr)[N]) noexcept;
// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   â€” extent == dynamic_extent || N == extent is true, and
//   â€” remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//

#include <uscl/std/cassert>
#include <uscl/std/span>

#include "test_macros.h"

__device__ int arr[]                  = {1, 2, 3};
__device__ const int carr[]           = {4, 5, 6};
__device__ volatile int varr[]        = {7, 8, 9};
__device__ const volatile int cvarr[] = {1, 3, 5};

int main(int, char**)
{
  //  Size wrong
  {
    cuda::std::span<int, 2> s1(arr); // expected-error {{no matching constructor for initialization of
                                     // 'cuda::std::span<int, 2>'}}
  }

  //  Type wrong
  {
    cuda::std::span<float> s1(arr); // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<float>'}}
    cuda::std::span<float, 3> s2(arr); // expected-error {{no matching constructor for initialization of
                                       // 'cuda::std::span<float, 3>'}}
  }

  //  CV wrong (dynamically sized)
  {
    cuda::std::span<int> s1{carr}; // expected-error {{no matching constructor for initialization of
                                   // 'cuda::std::span<int>'}}
    cuda::std::span<int> s2{varr}; // expected-error {{no matching constructor for initialization of
                                   // 'cuda::std::span<int>'}}
    cuda::std::span<int> s3{cvarr}; // expected-error {{no matching constructor for initialization of
                                    // 'cuda::std::span<int>'}}
    cuda::std::span<const int> s4{varr}; // expected-error {{no matching constructor for initialization of
                                         // 'cuda::std::span<const int>'}}
    cuda::std::span<const int> s5{cvarr}; // expected-error {{no matching constructor for initialization of
                                          // 'cuda::std::span<const int>'}}
    cuda::std::span<volatile int> s6{carr}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<volatile int>'}}
    cuda::std::span<volatile int> s7{cvarr}; // expected-error {{no matching constructor for initialization of
                                             // 'cuda::std::span<volatile int>'}}
  }

  //  CV wrong (statically sized)
  {
    cuda::std::span<int, 3> s1{carr}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<int, 3>'}}
    cuda::std::span<int, 3> s2{varr}; // expected-error {{no matching constructor for initialization of
                                      // 'cuda::std::span<int, 3>'}}
    cuda::std::span<int, 3> s3{cvarr}; // expected-error {{no matching constructor for initialization of
                                       // 'cuda::std::span<int, 3>'}}
    cuda::std::span<const int, 3> s4{varr}; // expected-error {{no matching constructor for initialization of
                                            // 'cuda::std::span<const int, 3>'}}
    cuda::std::span<const int, 3> s5{cvarr}; // expected-error {{no matching constructor for initialization of
                                             // 'cuda::std::span<const int, 3>'}}
    cuda::std::span<volatile int, 3> s6{carr}; // expected-error {{no matching constructor for initialization of
                                               // 'cuda::std::span<volatile int, 3>'}}
    cuda::std::span<volatile int, 3> s7{cvarr}; // expected-error {{no matching constructor for initialization of
                                                // 'cuda::std::span<volatile int, 3>'}}
  }

  return 0;
}
