/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 29, 2025.
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

// template <class ElementType, size_t Extent>
//     span<byte,
//          Extent == dynamic_extent
//              ? dynamic_extent
//              : sizeof(ElementType) * Extent>
//     as_writable_bytes(span<ElementType, Extent> s) noexcept;

#include <uscl/std/span>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int iArr2[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

struct A
{};

__host__ __device__ void f()
{
  cuda::std::as_writable_bytes(cuda::std::span<const int>()); // expected-error {{no matching function for call to
                                                              // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const long>()); // expected-error {{no matching function for call to
                                                               // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const double>()); // expected-error {{no matching function for call to
                                                                 // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const A>()); // expected-error {{no matching function for call to
                                                            // 'as_writable_bytes'}}

  cuda::std::as_writable_bytes(cuda::std::span<const int, 0>()); // expected-error {{no matching function for call to
                                                                 // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const long, 0>()); // expected-error {{no matching function for call to
                                                                  // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const double, 0>()); // expected-error {{no matching function for call to
                                                                    // 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const A, 0>()); // expected-error {{no matching function for call to
                                                               // 'as_writable_bytes'}}

  cuda::std::as_writable_bytes(cuda::std::span<const int>(iArr2, 1)); // expected-error {{no matching function for call
                                                                      // to 'as_writable_bytes'}}
  cuda::std::as_writable_bytes(cuda::std::span<const int, 1>(iArr2 + 5, 1)); // expected-error {{no matching function
                                                                             // for call to 'as_writable_bytes'}}
}

int main(int, char**)
{
  return 0;
}
