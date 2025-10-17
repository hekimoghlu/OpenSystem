/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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

// cuda::mr::resource_with

#include <uscl/memory_resource>
#include <uscl/std/cstdint>

struct prop_with_value
{};
struct prop
{};

struct valid_resource_with_property
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const valid_resource_with_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_with_property&) const
  {
    return false;
  }
  friend void get_property(const valid_resource_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::resource_with<valid_resource_with_property, prop_with_value>, "");

struct valid_resource_without_property
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const valid_resource_without_property&) const
  {
    return true;
  }
  bool operator!=(const valid_resource_without_property&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource_with<valid_resource_without_property, prop_with_value>, "");

struct invalid_resource_with_property
{
  friend void get_property(const invalid_resource_with_property&, prop_with_value) {}
};
static_assert(!cuda::mr::resource_with<invalid_resource_with_property, prop_with_value>, "");

struct resource_with_many_properties
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const resource_with_many_properties&) const
  {
    return true;
  }
  bool operator!=(const resource_with_many_properties&) const
  {
    return false;
  }
  friend void get_property(const resource_with_many_properties&, prop_with_value) {}
  friend void get_property(const resource_with_many_properties&, prop) {}
};
static_assert(cuda::mr::resource_with<resource_with_many_properties, prop_with_value, prop>, "");
static_assert(!cuda::mr::resource_with<resource_with_many_properties, prop_with_value, int, prop>, "");

struct derived_with_property : public valid_resource_without_property
{
  friend void get_property(const derived_with_property&, prop_with_value) {}
};
static_assert(cuda::mr::resource_with<derived_with_property, prop_with_value>, "");

int main(int, char**)
{
  return 0;
}
