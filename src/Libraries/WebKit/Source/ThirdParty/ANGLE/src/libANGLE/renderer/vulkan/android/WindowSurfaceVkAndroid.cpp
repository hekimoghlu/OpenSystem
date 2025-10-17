/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WindowSurfaceVkAndroid.cpp:
//    Implements the class methods for WindowSurfaceVkAndroid.
//

#include "libANGLE/renderer/vulkan/android/WindowSurfaceVkAndroid.h"

#include <android/native_window.h>

#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

WindowSurfaceVkAndroid::WindowSurfaceVkAndroid(const egl::SurfaceState &surfaceState,
                                               EGLNativeWindowType window)
    : WindowSurfaceVk(surfaceState, window)
{}

angle::Result WindowSurfaceVkAndroid::createSurfaceVk(vk::ErrorContext *context,
                                                      gl::Extents *extentsOut)
{
    VkAndroidSurfaceCreateInfoKHR createInfo = {};

    createInfo.sType  = VK_STRUCTURE_TYPE_ANDROID_SURFACE_CREATE_INFO_KHR;
    createInfo.flags  = 0;
    createInfo.window = mNativeWindowType;
    ANGLE_VK_TRY(context, vkCreateAndroidSurfaceKHR(context->getRenderer()->getInstance(),
                                                    &createInfo, nullptr, &mSurface));

    return getCurrentWindowSize(context, extentsOut);
}

angle::Result WindowSurfaceVkAndroid::getCurrentWindowSize(vk::ErrorContext *context,
                                                           gl::Extents *extentsOut)
{
    vk::Renderer *renderer                 = context->getRenderer();
    const VkPhysicalDevice &physicalDevice = renderer->getPhysicalDevice();
    VkSurfaceCapabilitiesKHR surfaceCaps;
    ANGLE_VK_TRY(context,
                 vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, mSurface, &surfaceCaps));
    *extentsOut = gl::Extents(surfaceCaps.currentExtent.width, surfaceCaps.currentExtent.height, 1);
    return angle::Result::Continue;
}

}  // namespace rx
