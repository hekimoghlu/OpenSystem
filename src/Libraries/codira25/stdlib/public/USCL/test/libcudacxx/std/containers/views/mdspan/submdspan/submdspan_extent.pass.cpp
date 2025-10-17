/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 11, 2023.
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

// UNSUPPORTED: c++11, c++14
// UNSUPPORTED: nvcc-12.0

#include <uscl/std/cassert>
#include <uscl/std/mdspan>
#include <uscl/std/type_traits>

#include "test_macros.h"

template <class Extent, class... Slices>
_CCCL_CONCEPT can_submdspan_extents = _CCCL_REQUIRES_EXPR(
  (Extent, variadic Slices), const Extent& ext, Slices... slices)((cuda::std::submdspan_extents(ext, slices...)));

__host__ __device__ constexpr bool test()
{
  { // single dimension, all static
    cuda::std::extents<size_t, 3> ext{};

    using extents_t = decltype(ext);
    static_assert(extents_t::rank() == 1);
    static_assert(extents_t::rank_dynamic() == 0);
    assert(ext.extent(0) == 3);
    assert(ext.static_extent(0) == 3);

    { // [mdspan.sub.extents-4.1]
      // S_k convertible_to<IndexType>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == 0);
      static_assert(subextents_t::rank_dynamic() == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.1]
      // S_k is_convertible<full_extent>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, cuda::std::full_extent);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.static_extent(0) == ext.static_extent(0));
      assert(sub_ext.extent(0) == ext.extent(0));
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and both model integral-constant-like
      const auto slice =
        cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, 2};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() + 1);
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{1, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() + 1);
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 0>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is not zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 2>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() + 1);
      assert(sub_ext.extent(0) == 2);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like but
      // extent is zero
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 0>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 2>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // Constraints: sizeof...(slices) == Extents::rank
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>>);
      static_assert(can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t>);
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t, size_t>);
    }
  }

  { // single dimension, all dynamic
    cuda::std::extents<size_t, cuda::std::dynamic_extent> ext{3};

    using extents_t = decltype(ext);
    static_assert(extents_t::rank() == 1);
    static_assert(extents_t::rank_dynamic() == 1);
    assert(ext.extent(0) == 3);
    assert(ext.static_extent(0) == cuda::std::dynamic_extent);

    { // [mdspan.sub.extents-4.1]
      // S_k convertible_to<IndexType>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == 0);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.1]
      // S_k is_convertible<full_extent>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, cuda::std::full_extent);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.static_extent(0) == ext.static_extent(0));
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and both model integral-constant-like
      const auto slice =
        cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, 2};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{1, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 0>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is not zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 2>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 2);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like but
      // extent is zero
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 0>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 2>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, slice);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank());
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // Constraints: sizeof...(slices) == Extents::rank
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>>);
      static_assert(can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t>);
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t, size_t>);
    }
  }

  { // multi dimension, mixed
    cuda::std::extents<size_t, 2, cuda::std::dynamic_extent, 4> ext{3};

    using extents_t = decltype(ext);
    static_assert(extents_t::rank() == 3);
    static_assert(extents_t::rank_dynamic() == 1);
    assert(ext.extent(0) == 2);
    assert(ext.extent(1) == 3);
    assert(ext.extent(2) == 4);
    assert(ext.static_extent(0) == 2);
    assert(ext.static_extent(1) == cuda::std::dynamic_extent);
    assert(ext.static_extent(2) == 4);

    { // [mdspan.sub.extents-4.1]
      // S_k convertible_to<IndexType>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, 2, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == 0);
      static_assert(subextents_t::rank_dynamic() == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.1]
      // S_k is_convertible<full_extent>
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, cuda::std::full_extent, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == 1);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == ext.extent(1));
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and both model integral-constant-like
      const auto slice =
        cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == 1);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.2]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{cuda::std::integral_constant<size_t, 1>{}, 2};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k models index-pair-like<IndexType> and one does not model integral-constant-like
      const auto slice           = cuda::std::pair{1, cuda::std::integral_constant<size_t, 2>{}};
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 0>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.3]
      // S_k is a specialization of strided_slice and extent_type models integral-constant-like and is not zero
      const auto slice = cuda::std::strided_slice{0, cuda::std::integral_constant<size_t, 2>{}, 1};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic());
      assert(sub_ext.extent(0) == 2);
      assert(sub_ext.static_extent(0) == cuda::std::dynamic_extent);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like but
      // extent is zero
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 0>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 0);
      assert(sub_ext.static_extent(0) == 0);
      unused(sub_ext);
    }

    { // [mdspan.sub.extents-4.2.4]
      // S_k is a specialization of strided_slice and extent_type and stride_type model integral-constant-like
      const auto slice = cuda::std::strided_slice{
        0, cuda::std::integral_constant<size_t, 2>{}, cuda::std::integral_constant<size_t, 2>{}};
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::extent_type>);
      static_assert(cuda::std::__integral_constant_like<typename decltype(slice)::stride_type>);
      cuda::std::extents sub_ext = cuda::std::submdspan_extents(ext, 1, slice, 1);

      using subextents_t = decltype(sub_ext);
      static_assert(subextents_t::rank() == extents_t::rank() - 2);
      static_assert(subextents_t::rank_dynamic() == extents_t::rank_dynamic() - 1);
      // [mdspan.sub.extents-5.1]
      // S_k.extent == 0 ? 0 : 1 + (de-ice(S_k.extent) - 1) / de-ice(S_k.stride)
      assert(sub_ext.extent(0) == 1);
      assert(sub_ext.static_extent(0) == 1);
      unused(sub_ext);
    }

    { // Constraints: sizeof...(slices) == Extents::rank
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>>);
      static_assert(can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t>);
      static_assert(!can_submdspan_extents<cuda::std::extents<size_t, 3>, size_t, size_t>);
    }
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
