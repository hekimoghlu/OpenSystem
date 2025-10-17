/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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
// Copyright 2021-2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WindowSurfaceVkWayland.cpp:
//    Implements the class methods for WindowSurfaceVkWayland.
//

#include "libANGLE/renderer/vulkan/linux/wayland/WindowSurfaceVkWayland.h"

#include "libANGLE/renderer/vulkan/vk_renderer.h"

#include <wayland-egl-backend.h>

namespace rx
{

void WindowSurfaceVkWayland::ResizeCallback(wl_egl_window *eglWindow, void *payload)
{
    WindowSurfaceVkWayland *windowSurface = reinterpret_cast<WindowSurfaceVkWayland *>(payload);

    windowSurface->mExtents.width  = eglWindow->width;
    windowSurface->mExtents.height = eglWindow->height;
}

WindowSurfaceVkWayland::WindowSurfaceVkWayland(const egl::SurfaceState &surfaceState,
                                               EGLNativeWindowType window,
                                               wl_display *display)
    : WindowSurfaceVk(surfaceState, window), mWaylandDisplay(display)
{
    wl_egl_window *eglWindow   = reinterpret_cast<wl_egl_window *>(window);
    eglWindow->resize_callback = WindowSurfaceVkWayland::ResizeCallback;
    eglWindow->driver_private  = this;

    mExtents = gl::Extents(eglWindow->width, eglWindow->height, 1);
}

angle::Result WindowSurfaceVkWayland::createSurfaceVk(vk::ErrorContext *context,
                                                      gl::Extents *extentsOut)
{
    ANGLE_VK_CHECK(context,
                   vkGetPhysicalDeviceWaylandPresentationSupportKHR(
                       context->getRenderer()->getPhysicalDevice(),
                       context->getRenderer()->getQueueFamilyIndex(), mWaylandDisplay),
                   VK_ERROR_INITIALIZATION_FAILED);

    wl_egl_window *eglWindow = reinterpret_cast<wl_egl_window *>(mNativeWindowType);

    VkWaylandSurfaceCreateInfoKHR createInfo = {};

    createInfo.sType   = VK_STRUCTURE_TYPE_WAYLAND_SURFACE_CREATE_INFO_KHR;
    createInfo.flags   = 0;
    createInfo.display = mWaylandDisplay;
    createInfo.surface = eglWindow->surface;
    ANGLE_VK_TRY(context, vkCreateWaylandSurfaceKHR(context->getRenderer()->getInstance(),
                                                    &createInfo, nullptr, &mSurface));

    return getCurrentWindowSize(context, extentsOut);
}

angle::Result WindowSurfaceVkWayland::getCurrentWindowSize(vk::ErrorContext *context,
                                                           gl::Extents *extentsOut)
{
    *extentsOut = mExtents;
    return angle::Result::Continue;
}

egl::Error WindowSurfaceVkWayland::getUserWidth(const egl::Display *display, EGLint *value) const
{
    *value = getWidth();
    return egl::NoError();
}

egl::Error WindowSurfaceVkWayland::getUserHeight(const egl::Display *display, EGLint *value) const
{
    *value = getHeight();
    return egl::NoError();
}

}  // namespace rx
