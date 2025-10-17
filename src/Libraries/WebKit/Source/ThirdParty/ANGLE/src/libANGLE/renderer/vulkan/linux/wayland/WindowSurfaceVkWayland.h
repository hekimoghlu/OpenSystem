/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
// WindowSurfaceVkWayland.h:
//    Defines the class interface for WindowSurfaceVkWayland, implementing WindowSurfaceVk.
//

#ifndef LIBANGLE_RENDERER_VULKAN_WAYLAND_WINDOWSURFACEVKWAYLAND_H_
#define LIBANGLE_RENDERER_VULKAN_WAYLAND_WINDOWSURFACEVKWAYLAND_H_

#include "libANGLE/renderer/vulkan/SurfaceVk.h"

struct wl_display;
struct wl_egl_window;

namespace rx
{

class WindowSurfaceVkWayland : public WindowSurfaceVk
{
  public:
    // Requests of new sizes from client go through this callback, but actual resize will happen
    // before the next operation which would provoke a backbuffer to be pulled.
    static void ResizeCallback(wl_egl_window *window, void *payload);

    WindowSurfaceVkWayland(const egl::SurfaceState &surfaceState,
                           EGLNativeWindowType window,
                           wl_display *display);

    // On Wayland, currentExtent is undefined (0xFFFFFFFF, 0xFFFFFFFF).
    // Whatever the application sets a swapchain's imageExtent to will be the size of the window,
    // after the first image is presented
    egl::Error getUserWidth(const egl::Display *display, EGLint *value) const override;
    egl::Error getUserHeight(const egl::Display *display, EGLint *value) const override;

  private:
    angle::Result createSurfaceVk(vk::ErrorContext *context, gl::Extents *extentsOut) override;
    angle::Result getCurrentWindowSize(vk::ErrorContext *context, gl::Extents *extentsOut) override;

    wl_display *mWaylandDisplay;
    gl::Extents mExtents;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_WAYLAND_WINDOWSURFACEVKWAYLAND_H_
