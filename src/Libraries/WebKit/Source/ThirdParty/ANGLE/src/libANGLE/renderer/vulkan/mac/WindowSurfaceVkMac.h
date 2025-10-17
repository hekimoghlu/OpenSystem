/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 9, 2023.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// WindowSurfaceVkMac.h:
//    Subclasses WindowSurfaceVk for the Mac platform.
//

#ifndef LIBANGLE_RENDERER_VULKAN_MAC_WINDOWSURFACEVKMAC_H_
#define LIBANGLE_RENDERER_VULKAN_MAC_WINDOWSURFACEVKMAC_H_

#include "libANGLE/renderer/vulkan/SurfaceVk.h"

#include <Cocoa/Cocoa.h>

namespace rx
{

class WindowSurfaceVkMac : public WindowSurfaceVk
{
  public:
    WindowSurfaceVkMac(const egl::SurfaceState &surfaceState, EGLNativeWindowType window);
    ~WindowSurfaceVkMac() override;

  private:
    angle::Result createSurfaceVk(vk::ErrorContext *context, gl::Extents *extentsOut) override;
    angle::Result getCurrentWindowSize(vk::ErrorContext *context, gl::Extents *extentsOut) override;

    CAMetalLayer *mMetalLayer;
    id<MTLDevice> mMetalDevice;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_MAC_WINDOWSURFACEVKMAC_H_
