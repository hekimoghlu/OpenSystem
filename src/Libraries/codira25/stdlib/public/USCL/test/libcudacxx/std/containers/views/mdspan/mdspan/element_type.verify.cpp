/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <mdspan>

// template<class ElementType, class Extents, class LayoutPolicy = layout_right, class AccessorPolicy =
// default_accessor> class mdspan;
//
// Mandates:
//   - ElementType is a complete object type that is neither an abstract class type nor an array type.
//   - is_same_v<ElementType, typename AccessorPolicy::element_type> is true.

#include <uscl/std/mdspan>

#include "test_macros.h"

class AbstractClass
{
public:
  __host__ __device__ virtual void method() = 0;
};

__host__ __device__ void not_abstract_class()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // may not be an abstract class}}
  cuda::std::mdspan<AbstractClass, cuda::std::extents<int>> m;
  unused(m);
}

__host__ __device__ void not_array_type()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // may not be an array type}}
  cuda::std::mdspan<int[5], cuda::std::extents<int>> m;
  unused(m);
}

__host__ __device__ void element_type_mismatch()
{
  // expected-error-re@*:* {{{{(static_assert|static assertion)}} failed {{.*}}mdspan: ElementType template parameter
  // must match AccessorPolicy::element_type}}
  cuda::std::mdspan<int, cuda::std::extents<int>, cuda::std::layout_right, cuda::std::default_accessor<const int>> m;
  unused(m);
}

int main(int, char**)
{
  return 0;
}
