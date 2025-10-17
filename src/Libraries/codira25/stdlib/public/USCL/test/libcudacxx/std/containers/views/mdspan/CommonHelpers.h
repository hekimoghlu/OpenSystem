/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H
#define TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H

#include <uscl/std/cassert>
#include <uscl/std/concepts>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class MDS, class H, cuda::std::enable_if_t<cuda::std::equality_comparable<H>, int> = 0>
__host__ __device__ constexpr void test_equality_handle(const MDS& m, const H& handle)
{
  assert(m.data_handle() == handle);
}
template <class MDS, class H, cuda::std::enable_if_t<!cuda::std::equality_comparable<H>, int> = 0>
__host__ __device__ constexpr void test_equality_handle(const MDS&, const H&)
{}

template <class MDS, class M, cuda::std::enable_if_t<cuda::std::equality_comparable<M>, int> = 0>
__host__ __device__ constexpr void test_equality_mapping(const MDS& m, const M& map)
{
  assert(m.mapping() == map);
}
template <class MDS, class M, cuda::std::enable_if_t<!cuda::std::equality_comparable<M>, int> = 0>
__host__ __device__ constexpr void test_equality_mapping(const MDS&, const M&)
{}

template <class MDS, class A, cuda::std::enable_if_t<cuda::std::equality_comparable<A>, int> = 0>
__host__ __device__ constexpr void test_equality_accessor(const MDS& m, const A& acc)
{
  assert(m.accessor() == acc);
}
template <class MDS, class A, cuda::std::enable_if_t<!cuda::std::equality_comparable<A>, int> = 0>
__host__ __device__ constexpr void test_equality_accessor(const MDS&, const A&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::data_handle_type, typename FromMDS::data_handle_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_handle(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.data_handle() == from_mds.data_handle());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::data_handle_type, typename FromMDS::data_handle_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_handle(const ToMDS&, const FromMDS&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::mapping_type, typename FromMDS::mapping_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_mapping(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.mapping() == from_mds.mapping());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::mapping_type, typename FromMDS::mapping_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_mapping(const ToMDS&, const FromMDS&)
{}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            cuda::std::equality_comparable_with<typename ToMDS::accessor_type, typename FromMDS::accessor_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_accessor(const ToMDS& to_mds, const FromMDS& from_mds)
{
  assert(to_mds.accessor() == from_mds.accessor());
}

template <class ToMDS,
          class FromMDS,
          cuda::std::enable_if_t<
            !cuda::std::equality_comparable_with<typename ToMDS::accessor_type, typename FromMDS::accessor_type>,
            int> = 0>
__host__ __device__ constexpr void test_equality_with_accessor(const ToMDS&, const FromMDS&)
{}

#endif // TEST_STD_CONTAINERS_VIEWS_MDSPAN_COMMON_HELPERS_TYPE_H
