/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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
// WindowSurfaceVkHeadless.cpp:
//    Implements the class methods for WindowSurfaceVkHeadless.
//

#include "WindowSurfaceVkHeadless.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

WindowSurfaceVkHeadless::WindowSurfaceVkHeadless(const egl::SurfaceState &surfaceState,
                                                 EGLNativeWindowType window)
    : WindowSurfaceVk(surfaceState, window)
{}

WindowSurfaceVkHeadless::~WindowSurfaceVkHeadless() {}

angle::Result WindowSurfaceVkHeadless::createSurfaceVk(vk::ErrorContext *context,
                                                       gl::Extents *extentsOut)
{
    vk::Renderer *renderer = context->getRenderer();
    ASSERT(renderer != nullptr);
    VkInstance instance = renderer->getInstance();

    VkHeadlessSurfaceCreateInfoEXT createInfo = {};
    createInfo.sType                          = VK_STRUCTURE_TYPE_HEADLESS_SURFACE_CREATE_INFO_EXT;

    ANGLE_VK_TRY(context, vkCreateHeadlessSurfaceEXT(instance, &createInfo, nullptr, &mSurface));

    return getCurrentWindowSize(context, extentsOut);
}

angle::Result WindowSurfaceVkHeadless::getCurrentWindowSize(vk::ErrorContext *context,
                                                            gl::Extents *extentsOut)
{
    const VkPhysicalDevice &physicalDevice = context->getRenderer()->getPhysicalDevice();
    ANGLE_VK_TRY(context, vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, mSurface,
                                                                    &mSurfaceCaps));

    // Spec: "For headless surfaces, currentExtent is the reserved value (0xFFFFFFFF, 0xFFFFFFFF).
    // Whatever the application sets a swapchain's imageExtent to will be the size of the surface,
    // after the first image is presented."
    // For ANGLE, in headless mode, we share the same 'SimpleDisplayWindow' structure with front
    // EGL window info to define the vulkan backend surface/image extents.
    angle::vk::SimpleDisplayWindow *simpleWindow =
        reinterpret_cast<angle::vk::SimpleDisplayWindow *>(mNativeWindowType);

    // Update surface extent before output the new extent.
    mSurfaceCaps.currentExtent.width  = simpleWindow->width;
    mSurfaceCaps.currentExtent.height = simpleWindow->height;

    *extentsOut =
        gl::Extents(mSurfaceCaps.currentExtent.width, mSurfaceCaps.currentExtent.height, 1);

    return angle::Result::Continue;
}

}  // namespace rx
