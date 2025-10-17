/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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

// cuda::mr::async_resource

#include <uscl/memory_resource>
#include <uscl/std/cstdint>

#include "test_macros.h"

struct invalid_argument
{};

struct valid_resource
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
  bool operator==(const valid_resource&) const
  {
    return true;
  }
  bool operator!=(const valid_resource&) const
  {
    return false;
  }
};
static_assert(cuda::mr::resource<valid_resource>, "");

struct invalid_allocate_missing
{
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_missing&) const
  {
    return true;
  }
  bool operator!=(const invalid_allocate_missing&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_missing>, "");

struct invalid_deallocate_missing
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_missing&) const
  {
    return true;
  }
  bool operator!=(const invalid_allocate_missing&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_deallocate_missing>, "");

struct invalid_allocate_async_argument
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  void* allocate(std::size_t, invalid_argument)
  {
    return nullptr;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_async_argument&) const
  {
    return true;
  }
  bool operator!=(const invalid_allocate_async_argument&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_async_argument>, "");

struct invalid_allocate_async_return
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  int allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return 42;
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) {}
  bool operator==(const invalid_allocate_async_return&) const
  {
    return true;
  }
  bool operator!=(const invalid_allocate_async_return&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_allocate_async_return>, "");

struct invalid_deallocate_async_argument
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
  void deallocate(std::size_t, void*, invalid_argument) {}
  bool operator==(const invalid_deallocate_async_argument&) const
  {
    return true;
  }
  bool operator!=(const invalid_deallocate_async_argument&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<invalid_deallocate_async_argument>, "");

struct non_comparable
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
};
static_assert(!cuda::mr::resource<non_comparable>, "");

struct non_eq_comparable
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
  bool operator!=(const non_eq_comparable&) const
  {
    return false;
  }
};
static_assert(!cuda::mr::resource<non_eq_comparable>, "");

#if TEST_STD_VER < 2020
struct non_neq_comparable
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
  bool operator==(const non_neq_comparable&) const
  {
    return true;
  }
};
static_assert(!cuda::mr::resource<non_neq_comparable>, "");
#endif // TEST_STD_VER < 2020

int main(int, char**)
{
  return 0;
}
