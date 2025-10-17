/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 25, 2025.
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

// <cuda/std/optional>

// T shall be an object type other than cv in_place_t or cv nullopt_t
//   and shall satisfy the Cpp17Destructible requirements.
// Note: array types do not satisfy the Cpp17Destructible requirements.

#include <uscl/std/cassert>
#include <uscl/std/optional>
#include <uscl/std/type_traits>

#include "test_macros.h"

struct NonDestructible
{
  ~NonDestructible() = delete;
};

int main(int, char**)
{
  {
    cuda::std::optional<char&> o1; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                   // failed{{.*}}instantiation of optional with a reference type is ill-formed}}
    cuda::std::optional<NonDestructible> o2; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                             // failed{{.*}}instantiation of optional with a non-destructible type is
                                             // ill-formed}}
    cuda::std::optional<char[20]> o3; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                      // failed{{.*}}instantiation of optional with an array type is ill-formed}}
  }

  {
    cuda::std::optional<cuda::std::in_place_t> o1; // expected-error-re@optional:* {{{{(static_assert|static
                                                   // assertion)}} failed{{.*}}instantiation of optional with in_place_t
                                                   // is ill-formed}}
    cuda::std::optional<const cuda::std::in_place_t> o2; // expected-error-re@optional:* {{{{(static_assert|static
                                                         // assertion)}} failed{{.*}}instantiation of optional with
                                                         // in_place_t is ill-formed}}
    cuda::std::optional<volatile cuda::std::in_place_t> o3; // expected-error-re@optional:* {{{{(static_assert|static
                                                            // assertion)}} failed{{.*}}instantiation of optional with
                                                            // in_place_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::in_place_t> o4; // expected-error-re@optional:*
                                                                  // {{{{(static_assert|static assertion)}}
                                                                  // failed{{.*}}instantiation of optional with
                                                                  // in_place_t is ill-formed}}
  }

  {
    cuda::std::optional<cuda::std::nullopt_t> o1; // expected-error-re@optional:* {{{{(static_assert|static assertion)}}
                                                  // failed{{.*}}instantiation of optional with nullopt_t is
                                                  // ill-formed}}
    cuda::std::optional<const cuda::std::nullopt_t> o2; // expected-error-re@optional:* {{{{(static_assert|static
                                                        // assertion)}} failed{{.*}}instantiation of optional with
                                                        // nullopt_t is ill-formed}}
    cuda::std::optional<volatile cuda::std::nullopt_t> o3; // expected-error-re@optional:* {{{{(static_assert|static
                                                           // assertion)}} failed{{.*}}instantiation of optional with
                                                           // nullopt_t is ill-formed}}
    cuda::std::optional<const volatile cuda::std::nullopt_t> o4; // expected-error-re@optional:*
                                                                 // {{{{(static_assert|static assertion)}}
                                                                 // failed{{.*}}instantiation of optional with nullopt_t
                                                                 // is ill-formed}}
  }

  return 0;
}
