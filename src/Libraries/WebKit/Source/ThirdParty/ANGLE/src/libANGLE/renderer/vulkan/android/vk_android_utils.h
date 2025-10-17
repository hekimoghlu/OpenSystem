/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 29, 2021.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// vk_android_utils.h: Vulkan utilities for using the Android platform

#ifndef LIBANGLE_RENDERER_VULKAN_ANDROID_VK_ANDROID_UTILS_H_
#define LIBANGLE_RENDERER_VULKAN_ANDROID_VK_ANDROID_UTILS_H_

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "common/vulkan/vk_headers.h"
#include "libANGLE/Error.h"

class Buffer;
class DeviceMemory;

namespace rx
{
class ErrorContext;

namespace vk
{
class Renderer;
angle::Result InitAndroidExternalMemory(ErrorContext *context,
                                        EGLClientBuffer clientBuffer,
                                        VkMemoryPropertyFlags memoryProperties,
                                        Buffer *buffer,
                                        VkMemoryPropertyFlags *memoryPropertyFlagsOut,
                                        uint32_t *memoryTypeIndexOut,
                                        DeviceMemory *deviceMemoryOut,
                                        VkDeviceSize *sizeOut);

void ReleaseAndroidExternalMemory(Renderer *renderer, EGLClientBuffer clientBuffer);
}  // namespace vk
}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_ANDROID_VK_ANDROID_UTILS_H_
