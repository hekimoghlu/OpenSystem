/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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

#include <uscl/memory_resource>
#include <uscl/std/cassert>
#include <uscl/std/type_traits>

// Verify that the properties exist
static_assert(cuda::std::is_empty<cuda::mr::host_accessible>::value, "");
static_assert(cuda::std::is_empty<cuda::mr::device_accessible>::value, "");

// Verify that host accessible is the default if nothing is specified
static_assert(!cuda::mr::__is_host_accessible<>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_device_accessible<>, "");
static_assert(!cuda::mr::__is_device_accessible<cuda::mr::host_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that host device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_host_device_accessible<>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

int main(int, char**)
{
  return 0;
}
