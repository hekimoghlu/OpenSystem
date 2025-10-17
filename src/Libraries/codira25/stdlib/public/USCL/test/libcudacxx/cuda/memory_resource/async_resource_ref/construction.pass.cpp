/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource_ref construction

#include <uscl/memory_resource>
#include <uscl/std/cstdint>
#include <uscl/std/type_traits>

#include "types.h"

namespace constructible
{
using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using matching_properties =
  test_resource<cuda::mr::host_accessible,
                property_with_value<double>,
                property_without_value<std::size_t>,
                property_with_value<int>>;

using missing_stateful_property =
  test_resource<cuda::mr::host_accessible, property_with_value<int>, property_without_value<std::size_t>>;
using missing_stateless_property =
  test_resource<cuda::mr::host_accessible, property_with_value<int>, property_with_value<double>>;

using cuda::std::is_constructible;
static_assert(is_constructible<ref, matching_properties&>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property&>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property&>::value, "");

static_assert(is_constructible<ref, matching_properties*>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property*>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property*>::value, "");

static_assert(is_constructible<ref, ref&>::value, "");

// Ensure we require a mutable valid reference and do not bind against rvalues
static_assert(!is_constructible<ref, matching_properties>::value, "");
static_assert(!is_constructible<ref, const matching_properties&>::value, "");
static_assert(!is_constructible<ref, const matching_properties*>::value, "");

static_assert(cuda::std::is_copy_constructible<ref>::value, "");
static_assert(cuda::std::is_move_constructible<ref>::value, "");
} // namespace constructible

namespace assignable
{
using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using res = test_resource<cuda::mr::host_accessible,
                          property_with_value<int>,
                          property_with_value<double>,
                          property_without_value<std::size_t>>;

using other_res =
  test_resource<cuda::mr::host_accessible,
                property_without_value<int>,
                property_with_value<int>,
                property_with_value<double>,
                property_without_value<std::size_t>>;

using cuda::std::is_assignable;
static_assert(cuda::std::is_assignable<ref, res&>::value, "");
static_assert(cuda::std::is_assignable<ref, other_res&>::value, "");

static_assert(cuda::std::is_copy_assignable<ref>::value, "");
static_assert(cuda::std::is_move_assignable<ref>::value, "");
} // namespace assignable

int main(int, char**)
{
  return 0;
}
