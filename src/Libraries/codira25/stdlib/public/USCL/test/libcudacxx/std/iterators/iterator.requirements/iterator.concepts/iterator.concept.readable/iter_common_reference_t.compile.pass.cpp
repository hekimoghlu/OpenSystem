/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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

// iter_common_reference_t

#include <uscl/std/concepts>
#include <uscl/std/iterator>

struct X
{};

// value_type and dereferencing are the same
struct T1
{
  using value_type = X;
  __host__ __device__ X operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T1>, X>, "");

// value_type and dereferencing are the same (modulo qualifiers)
struct T2
{
  using value_type = X;
  __host__ __device__ X& operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T2>, X&>, "");

// There's a custom common reference between value_type and the type of dereferencing
struct A
{};
struct B
{};
struct Common
{
  __host__ __device__ Common(A);
  __host__ __device__ Common(B);
};

namespace cuda
{
namespace std
{
template <template <class> class TQual, template <class> class QQual>
struct basic_common_reference<A, B, TQual, QQual>
{
  using type = Common;
};
template <template <class> class TQual, template <class> class QQual>
struct basic_common_reference<B, A, TQual, QQual> : basic_common_reference<A, B, TQual, QQual>
{};
} // namespace std
} // namespace cuda

struct T3
{
  using value_type = A;
  __host__ __device__ B&& operator*() const;
};
static_assert(cuda::std::same_as<cuda::std::iter_common_reference_t<T3>, Common>, "");

// Make sure we're SFINAE-friendly
template <class T>
_CCCL_CONCEPT has_common_reference = _CCCL_REQUIRES_EXPR((T))(typename(cuda::std::iter_common_reference_t<T>));

struct NotIndirectlyReadable
{};
static_assert(!has_common_reference<NotIndirectlyReadable>, "");

int main(int, char**)
{
  return 0;
}
