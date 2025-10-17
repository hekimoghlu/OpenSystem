/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 10, 2025.
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
// Copyright 2024 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WindowSurfaceVkWin32.cpp:
//    Defines the class interface for WindowSurfaceWgpuWin32, implementing WindowSurfaceWgpu.
//

#include "libANGLE/renderer/wgpu/win32/WindowSurfaceWgpuWin32.h"

#include "libANGLE/Display.h"
#include "libANGLE/renderer/wgpu/DisplayWgpu.h"
#include "libANGLE/renderer/wgpu/wgpu_utils.h"

namespace rx
{
WindowSurfaceWgpuWin32::WindowSurfaceWgpuWin32(const egl::SurfaceState &surfaceState,
                                               EGLNativeWindowType window)
    : WindowSurfaceWgpu(surfaceState, window)
{}

angle::Result WindowSurfaceWgpuWin32::createWgpuSurface(const egl::Display *display,
                                                        wgpu::Surface *outSurface)
{
    DisplayWgpu *displayWgpu = webgpu::GetImpl(display);
    auto &surfaceCache       = displayWgpu->getSurfaceCache();

    EGLNativeWindowType window = getNativeWindow();
    auto cachedSurfaceIter     = surfaceCache.find(window);
    if (cachedSurfaceIter != surfaceCache.end())
    {
        *outSurface = cachedSurfaceIter->second;
        return angle::Result::Continue;
    }

    wgpu::Instance instance = displayWgpu->getInstance();

    wgpu::SurfaceDescriptorFromWindowsHWND hwndDesc;
    hwndDesc.hinstance = GetModuleHandle(nullptr);
    hwndDesc.hwnd      = window;

    wgpu::SurfaceDescriptor surfaceDesc;
    surfaceDesc.nextInChain = &hwndDesc;

    wgpu::Surface surface = instance.CreateSurface(&surfaceDesc);
    *outSurface           = surface;

    surfaceCache.insert_or_assign(window, surface);
    return angle::Result::Continue;
}

angle::Result WindowSurfaceWgpuWin32::getCurrentWindowSize(const egl::Display *display,
                                                           gl::Extents *outSize)
{
    RECT rect;
    if (!GetClientRect(getNativeWindow(), &rect))
    {
        // TODO: generate a proper error + msg
        return angle::Result::Stop;
    }

    *outSize = gl::Extents(rect.right - rect.left, rect.bottom - rect.top, 1);
    return angle::Result::Continue;
}

WindowSurfaceWgpu *CreateWgpuWindowSurface(const egl::SurfaceState &surfaceState,
                                           EGLNativeWindowType window)
{
    return new WindowSurfaceWgpuWin32(surfaceState, window);
}
}  // namespace rx
