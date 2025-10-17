/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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
// WindowSurfaceVkWin32.h:
//    Defines the class interface for WindowSurfaceWgpuWin32, implementing WindowSurfaceWgpu.
//

#ifndef LIBANGLE_RENDERER_WGPU_WIN32_WINDOWSURFACEWGPUWIN32_H_
#define LIBANGLE_RENDERER_WGPU_WIN32_WINDOWSURFACEWGPUWIN32_H_

#include "libANGLE/renderer/wgpu/SurfaceWgpu.h"

namespace rx
{
class WindowSurfaceWgpuWin32 : public WindowSurfaceWgpu
{
  public:
    WindowSurfaceWgpuWin32(const egl::SurfaceState &surfaceState, EGLNativeWindowType window);

  private:
    angle::Result createWgpuSurface(const egl::Display *display,
                                    wgpu::Surface *outSurface) override;
    angle::Result getCurrentWindowSize(const egl::Display *display, gl::Extents *outSize) override;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_WIN32_WINDOWSURFACEWGPUWIN32_H_
