/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 6, 2023.
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
// WindowSurfaceWgpuMetalLayer.h:
//    Defines the class interface for WindowSurfaceWgpuMetalLayer, implementing WindowSurfaceWgpu.
//

#ifndef LIBANGLE_RENDERER_WGPU_MAC_WINDOWSURFACEWGPUMETALLAYER_H_
#define LIBANGLE_RENDERER_WGPU_MAC_WINDOWSURFACEWGPUMETALLAYER_H_

#include "libANGLE/renderer/wgpu/SurfaceWgpu.h"

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

namespace rx
{
class WindowSurfaceWgpuMetalLayer : public WindowSurfaceWgpu
{
  public:
    WindowSurfaceWgpuMetalLayer(const egl::SurfaceState &surfaceState, EGLNativeWindowType window);

    egl::Error initialize(const egl::Display *display) override;
    void destroy(const egl::Display *display) override;

  private:
    angle::Result createWgpuSurface(const egl::Display *display,
                                    wgpu::Surface *outSurface) override;
    angle::Result getCurrentWindowSize(const egl::Display *display, gl::Extents *outSize) override;

    id<MTLDevice> mMetalDevice;
    CAMetalLayer *mMetalLayer;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_MAC_WINDOWSURFACEWGPUMETALLAYER_H_
