/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WindowSurfaceVkWin32.cpp:
//    Implements the class methods for WindowSurfaceVkWin32.
//

#include "libANGLE/renderer/vulkan/win32/WindowSurfaceVkWin32.h"

#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

WindowSurfaceVkWin32::WindowSurfaceVkWin32(const egl::SurfaceState &surfaceState,
                                           EGLNativeWindowType window)
    : WindowSurfaceVk(surfaceState, window)
{}

angle::Result WindowSurfaceVkWin32::createSurfaceVk(vk::ErrorContext *context,
                                                    gl::Extents *extentsOut)
{
    VkWin32SurfaceCreateInfoKHR createInfo = {};

    createInfo.sType     = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    createInfo.flags     = 0;
    createInfo.hinstance = GetModuleHandle(nullptr);
    createInfo.hwnd      = mNativeWindowType;
    ANGLE_VK_TRY(context, vkCreateWin32SurfaceKHR(context->getRenderer()->getInstance(),
                                                  &createInfo, nullptr, &mSurface));

    return getCurrentWindowSize(context, extentsOut);
}

angle::Result WindowSurfaceVkWin32::getCurrentWindowSize(vk::ErrorContext *context,
                                                         gl::Extents *extentsOut)
{
    RECT rect;
    ANGLE_VK_CHECK(context, GetClientRect(mNativeWindowType, &rect) == TRUE,
                   VK_ERROR_INITIALIZATION_FAILED);

    *extentsOut = gl::Extents(rect.right - rect.left, rect.bottom - rect.top, 1);
    return angle::Result::Continue;
}

}  // namespace rx
