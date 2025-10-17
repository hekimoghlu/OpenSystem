/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
// WindowSurfaceWgpuMetalLayer.cpp:
//    Defines the class interface for WindowSurfaceWgpuMetalLayer, implementing WindowSurfaceWgpu.
//

#include "libANGLE/renderer/wgpu/mac/WindowSurfaceWgpuMetalLayer.h"

#include <Metal/Metal.h>
#include <QuartzCore/CAMetalLayer.h>

#include "libANGLE/Display.h"
#include "libANGLE/renderer/wgpu/DisplayWgpu.h"
#include "libANGLE/renderer/wgpu/wgpu_utils.h"

namespace rx
{

WindowSurfaceWgpuMetalLayer::WindowSurfaceWgpuMetalLayer(const egl::SurfaceState &surfaceState,
                                                         EGLNativeWindowType window)
    : WindowSurfaceWgpu(surfaceState, window)
{}

egl::Error WindowSurfaceWgpuMetalLayer::initialize(const egl::Display *display)
{
    // TODO: Use the same Metal device as wgpu
    mMetalDevice = MTLCreateSystemDefaultDevice();

    return WindowSurfaceWgpu::initialize(display);
}

void WindowSurfaceWgpuMetalLayer::destroy(const egl::Display *display)
{
    WindowSurfaceWgpu::destroy(display);
    [mMetalDevice release];
    if (mMetalLayer)
    {
        [mMetalLayer removeFromSuperlayer];
        [mMetalLayer release];
    }
}

angle::Result WindowSurfaceWgpuMetalLayer::createWgpuSurface(const egl::Display *display,
                                                             wgpu::Surface *outSurface)
    API_AVAILABLE(macosx(10.11))
{
    CALayer *layer = reinterpret_cast<CALayer *>(getNativeWindow());

    mMetalLayer        = [[CAMetalLayer alloc] init];
    mMetalLayer.frame  = CGRectMake(0, 0, layer.frame.size.width, layer.frame.size.height);
    mMetalLayer.device = mMetalDevice;
    mMetalLayer.drawableSize =
        CGSizeMake(mMetalLayer.bounds.size.width * mMetalLayer.contentsScale,
                   mMetalLayer.bounds.size.height * mMetalLayer.contentsScale);
    mMetalLayer.framebufferOnly  = NO;
    mMetalLayer.autoresizingMask = kCALayerWidthSizable | kCALayerHeightSizable;
    mMetalLayer.contentsScale    = layer.contentsScale;

    [layer addSublayer:mMetalLayer];

    wgpu::SurfaceDescriptorFromMetalLayer metalLayerDesc;
    metalLayerDesc.layer = mMetalLayer;

    wgpu::SurfaceDescriptor surfaceDesc;
    surfaceDesc.nextInChain = &metalLayerDesc;

    DisplayWgpu *displayWgpu = webgpu::GetImpl(display);
    wgpu::Instance instance  = displayWgpu->getInstance();

    wgpu::Surface surface = instance.CreateSurface(&surfaceDesc);
    *outSurface           = surface;

    return angle::Result::Continue;
}

angle::Result WindowSurfaceWgpuMetalLayer::getCurrentWindowSize(const egl::Display *display,
                                                                gl::Extents *outSize)
    API_AVAILABLE(macosx(10.11))
{
    ASSERT(mMetalLayer != nullptr);

    mMetalLayer.drawableSize =
        CGSizeMake(mMetalLayer.bounds.size.width * mMetalLayer.contentsScale,
                   mMetalLayer.bounds.size.height * mMetalLayer.contentsScale);
    *outSize = gl::Extents(static_cast<int>(mMetalLayer.drawableSize.width),
                           static_cast<int>(mMetalLayer.drawableSize.height), 1);

    return angle::Result::Continue;
}

WindowSurfaceWgpu *CreateWgpuWindowSurface(const egl::SurfaceState &surfaceState,
                                           EGLNativeWindowType window)
{
    return new WindowSurfaceWgpuMetalLayer(surfaceState, window);
}
}  // namespace rx
