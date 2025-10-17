/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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
// DisplayVkGGP.cpp:
//    Implements the class methods for DisplayVkGGP.
//

#include "libANGLE/renderer/vulkan/ggp/DisplayVkGGP.h"

#include "libANGLE/renderer/vulkan/ggp/WindowSurfaceVkGGP.h"
#include "libANGLE/renderer/vulkan/vk_caps_utils.h"

namespace rx
{
DisplayVkGGP::DisplayVkGGP(const egl::DisplayState &state) : DisplayVk(state) {}

bool DisplayVkGGP::isValidNativeWindow(EGLNativeWindowType window) const
{
    // GGP doesn't use window handles.
    return true;
}

SurfaceImpl *DisplayVkGGP::createWindowSurfaceVk(const egl::SurfaceState &state,
                                                 EGLNativeWindowType window)
{
    return new WindowSurfaceVkGGP(state, window);
}

egl::ConfigSet DisplayVkGGP::generateConfigs()
{
    // Not entirely sure what backbuffer formats GGP supports.
    constexpr GLenum kColorFormats[] = {GL_BGRA8_EXT, GL_BGRX8_ANGLEX};
    return egl_vk::GenerateConfigs(kColorFormats, egl_vk::kConfigDepthStencilFormats, this);
}

void DisplayVkGGP::checkConfigSupport(egl::Config *config) {}

const char *DisplayVkGGP::getWSIExtension() const
{
    return VK_GGP_STREAM_DESCRIPTOR_SURFACE_EXTENSION_NAME;
}

bool IsVulkanGGPDisplayAvailable()
{
    return true;
}

DisplayImpl *CreateVulkanGGPDisplay(const egl::DisplayState &state)
{
    return new DisplayVkGGP(state);
}
}  // namespace rx
