/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 12, 2024.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayVkNull.cpp:
//    Implements the class methods for DisplayVkNull.
//

#include "DisplayVkNull.h"

#include "libANGLE/Display.h"
#include "libANGLE/renderer/vulkan/SurfaceVk.h"
#include "libANGLE/renderer/vulkan/vk_caps_utils.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

DisplayVkNull::DisplayVkNull(const egl::DisplayState &state) : DisplayVk(state) {}

bool DisplayVkNull::isValidNativeWindow(EGLNativeWindowType window) const
{
    return false;
}

SurfaceImpl *DisplayVkNull::createWindowSurfaceVk(const egl::SurfaceState &state,
                                                  EGLNativeWindowType window)
{
    return new OffscreenSurfaceVk(state, mRenderer);
}

const char *DisplayVkNull::getWSIExtension() const
{
    return nullptr;
}

egl::ConfigSet DisplayVkNull::generateConfigs()
{
    constexpr GLenum kColorFormats[] = {GL_RGBA8, GL_BGRA8_EXT, GL_RGB565, GL_RGB8};

    return egl_vk::GenerateConfigs(kColorFormats, egl_vk::kConfigDepthStencilFormats, this);
}

void DisplayVkNull::checkConfigSupport(egl::Config *config) {}

bool IsVulkanNullDisplayAvailable()
{
    return true;
}

DisplayImpl *CreateVulkanNullDisplay(const egl::DisplayState &state)
{
    return new DisplayVkNull(state);
}

}  // namespace rx
