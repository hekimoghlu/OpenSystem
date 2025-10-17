/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 10, 2023.
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

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::has_property, cuda::has_property_with

#include <uscl/memory_resource>

struct prop_with_value
{
  using value_type = int;
};
struct prop
{};

static_assert(cuda::property_with_value<prop_with_value>, "");
static_assert(!cuda::property_with_value<prop>, "");

struct valid_property
{
  friend void get_property(const valid_property&, prop) {}
};
static_assert(!cuda::has_property<valid_property, prop_with_value>, "");
static_assert(cuda::has_property<valid_property, prop>, "");
static_assert(!cuda::has_property_with<valid_property, prop, int>, "");

struct valid_property_with_value
{
  friend int get_property(const valid_property_with_value&, prop_with_value)
  {
    return 42;
  }
};
static_assert(cuda::has_property<valid_property_with_value, prop_with_value>, "");
static_assert(!cuda::has_property<valid_property_with_value, prop>, "");
static_assert(cuda::has_property_with<valid_property_with_value, prop_with_value, int>, "");
static_assert(!cuda::has_property_with<valid_property_with_value, prop_with_value, double>, "");

struct derived_from_property : public valid_property
{
  friend int get_property(const derived_from_property&, prop_with_value)
  {
    return 42;
  }
};
static_assert(cuda::has_property<derived_from_property, prop_with_value>, "");
static_assert(cuda::has_property<derived_from_property, prop>, "");
static_assert(cuda::has_property_with<derived_from_property, prop_with_value, int>, "");
static_assert(!cuda::has_property_with<derived_from_property, prop_with_value, double>, "");

int main(int, char**)
{
  return 0;
}
