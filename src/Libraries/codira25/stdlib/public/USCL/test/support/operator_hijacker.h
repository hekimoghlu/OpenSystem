/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 11, 2021.
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

#ifndef SUPPORT_OPERATOR_HIJACKER_H
#define SUPPORT_OPERATOR_HIJACKER_H

#include <uscl/std/cstddef>
#include <uscl/std/functional>

#include "test_macros.h"

/// Helper struct to test ADL-hijacking in containers.
///
/// The class has some additional operations to be usable in all containers.
struct operator_hijacker
{
  __host__ __device__ bool operator<(const operator_hijacker&) const
  {
    return true;
  }
  __host__ __device__ bool operator==(const operator_hijacker&) const
  {
    return true;
  }

  template <typename T>
  __host__ __device__ friend void operator&(T&&) = delete;
  template <class T, class U>
  __host__ __device__ friend void operator,(T&&, U&&) = delete;
  template <class T, class U>
  __host__ __device__ friend void operator&&(T&&, U&&) = delete;
  template <class T, class U>
  __host__ __device__ friend void operator||(T&&, U&&) = delete;
};

static_assert(cuda::std::is_trivially_copyable<operator_hijacker>::value && //
                cuda::std::is_copy_constructible<operator_hijacker>::value && //
                cuda::std::is_move_constructible<operator_hijacker>::value && //
                cuda::std::is_copy_assignable<operator_hijacker>::value && //
                cuda::std::is_move_assignable<operator_hijacker>::value, //
              "does not satisfy the requirements for atomic<operator_hijacker>");
/*
template <>
struct cuda::std::hash<operator_hijacker> {
  cuda::std::size_t operator()(const operator_hijacker&) const { return 0; }
};
*/
#endif // SUPPORT_OPERATOR_HIJACKER_H
