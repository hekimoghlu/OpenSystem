/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 4, 2025.
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

#ifndef TEST_MEMORY_RESOURCE_ASYNC_RESOURCE_REF_TYPES_H
#define TEST_MEMORY_RESOURCE_ASYNC_RESOURCE_REF_TYPES_H

#include <uscl/memory_resource>

template <class T>
struct property_with_value
{
  using value_type = T;
};

template <class T>
struct property_without_value
{};

template <class... Properties>
struct test_resource
{
  void* allocate_sync(std::size_t, std::size_t)
  {
    return nullptr;
  }

  void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  void* allocate(cuda::stream_ref, std::size_t, std::size_t)
  {
    return &_val;
  }

  void deallocate(cuda::stream_ref, void* ptr, std::size_t, std::size_t)
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const test_resource& other) const
  {
    return _val == other._val;
  }
  bool operator!=(const test_resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES((!cuda::property_with_value<Property>) && ::cuda::std::__is_included_in_v<Property, Properties...>)
  friend void get_property(const test_resource&, Property) noexcept {}

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES(cuda::property_with_value<Property>&& ::cuda::std::__is_included_in_v<Property, Properties...>)
  friend typename Property::value_type get_property(const test_resource& res, Property) noexcept
  {
    return static_cast<typename Property::value_type>(res._val);
  }
};

#endif // TEST_MEMORY_RESOURCE_ASYNC_RESOURCE_REF_TYPES_H
