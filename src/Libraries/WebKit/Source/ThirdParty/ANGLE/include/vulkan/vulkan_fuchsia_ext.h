/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// vulkan_fuchsia_ext:
//   Defines Fuchsia-specific Vulkan extensions when compiling on other
//   platforms.
//

#ifndef COMMON_VULKAN_FUCHSIA_EXT_H_
#define COMMON_VULKAN_FUCHSIA_EXT_H_

#if !defined(VK_NO_PROTOTYPES)
#    define VK_NO_PROTOTYPES
#endif

#include <vulkan/vulkan.h>

// If this is not Fuchsia then define Fuchsia-specific types explicitly and include
// vulkan_fuchsia.h to make it possible to compile the code on other platforms.
//
// TODO(https://anglebug.com/42264570): Update all code to avoid dependencies on
// Fuchsia-specific types when compiling on other platforms. Then remove this header.
#if !defined(ANGLE_PLATFORM_FUCHSIA)
typedef uint32_t zx_handle_t;
#    define ZX_HANDLE_INVALID ((zx_handle_t)0)
#    include <vulkan/vulkan_fuchsia.h>
#endif

#endif  // COMMON_VULKAN_FUCHSIA_EXT_H_
