/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

// <ranges>

// template<class T>
// concept view = ...;

#include <uscl/std/ranges>

#include "test_macros.h"

// The type would be a view, but it's not moveable.
struct NotMoveable : cuda::std::ranges::view_base
{
  NotMoveable()                         = default;
  NotMoveable(NotMoveable&&)            = delete;
  NotMoveable& operator=(NotMoveable&&) = delete;
  __host__ __device__ friend int* begin(NotMoveable&);
  __host__ __device__ friend int* begin(NotMoveable const&);
  __host__ __device__ friend int* end(NotMoveable&);
  __host__ __device__ friend int* end(NotMoveable const&);
};
static_assert(cuda::std::ranges::range<NotMoveable>, "");
static_assert(!cuda::std::movable<NotMoveable>, "");
static_assert(cuda::std::default_initializable<NotMoveable>, "");
static_assert(cuda::std::ranges::enable_view<NotMoveable>, "");
static_assert(!cuda::std::ranges::view<NotMoveable>, "");

// The type would be a view, but it's not default initializable
struct NotDefaultInit : cuda::std::ranges::view_base
{
  NotDefaultInit() = delete;
  __host__ __device__ friend int* begin(NotDefaultInit&);
  __host__ __device__ friend int* begin(NotDefaultInit const&);
  __host__ __device__ friend int* end(NotDefaultInit&);
  __host__ __device__ friend int* end(NotDefaultInit const&);
};
static_assert(cuda::std::ranges::range<NotDefaultInit>, "");
static_assert(cuda::std::movable<NotDefaultInit>, "");
static_assert(!cuda::std::default_initializable<NotDefaultInit>, "");
static_assert(cuda::std::ranges::enable_view<NotDefaultInit>, "");
static_assert(cuda::std::ranges::view<NotDefaultInit>, "");

// The type would be a view, but it doesn't enable it with enable_view
struct NotExplicitlyEnabled
{
  NotExplicitlyEnabled()                                  = default;
  NotExplicitlyEnabled(NotExplicitlyEnabled&&)            = default;
  NotExplicitlyEnabled& operator=(NotExplicitlyEnabled&&) = default;
  __host__ __device__ friend int* begin(NotExplicitlyEnabled&);
  __host__ __device__ friend int* begin(NotExplicitlyEnabled const&);
  __host__ __device__ friend int* end(NotExplicitlyEnabled&);
  __host__ __device__ friend int* end(NotExplicitlyEnabled const&);
};
static_assert(cuda::std::ranges::range<NotExplicitlyEnabled>, "");
static_assert(cuda::std::movable<NotExplicitlyEnabled>, "");
static_assert(cuda::std::default_initializable<NotExplicitlyEnabled>, "");
static_assert(!cuda::std::ranges::enable_view<NotExplicitlyEnabled>, "");
static_assert(!cuda::std::ranges::view<NotExplicitlyEnabled>, "");

// The type has everything else, but it's not a range
struct NotARange : cuda::std::ranges::view_base
{
  NotARange()                       = default;
  NotARange(NotARange&&)            = default;
  NotARange& operator=(NotARange&&) = default;
};
static_assert(!cuda::std::ranges::range<NotARange>, "");
static_assert(cuda::std::movable<NotARange>, "");
static_assert(cuda::std::default_initializable<NotARange>, "");
static_assert(cuda::std::ranges::enable_view<NotARange>, "");
static_assert(!cuda::std::ranges::view<NotARange>, "");

// The type satisfies all requirements
struct View : cuda::std::ranges::view_base
{
  View()                  = default;
  View(View&&)            = default;
  View& operator=(View&&) = default;
  __host__ __device__ friend int* begin(View&);
  __host__ __device__ friend int* begin(View const&);
  __host__ __device__ friend int* end(View&);
  __host__ __device__ friend int* end(View const&);
};
static_assert(cuda::std::ranges::range<View>, "");
static_assert(cuda::std::movable<View>, "");
static_assert(cuda::std::default_initializable<View>, "");
static_assert(cuda::std::ranges::enable_view<View>, "");
static_assert(cuda::std::ranges::view<View>, "");

int main(int, char**)
{
  return 0;
}
