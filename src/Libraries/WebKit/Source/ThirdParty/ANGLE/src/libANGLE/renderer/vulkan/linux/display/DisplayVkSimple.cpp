/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 9, 2024.
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
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayVkSimple.cpp:
//    Implements the class methods for DisplayVkSimple.
//

#include "DisplayVkSimple.h"
#include "WindowSurfaceVkSimple.h"

#include "libANGLE/Display.h"
#include "libANGLE/renderer/vulkan/vk_caps_utils.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

DisplayVkSimple::DisplayVkSimple(const egl::DisplayState &state) : DisplayVkLinux(state) {}

void DisplayVkSimple::terminate()
{
    DisplayVk::terminate();
}

bool DisplayVkSimple::isValidNativeWindow(EGLNativeWindowType window) const
{
    return true;
}

SurfaceImpl *DisplayVkSimple::createWindowSurfaceVk(const egl::SurfaceState &state,
                                                    EGLNativeWindowType window)
{
    return new WindowSurfaceVkSimple(state, window);
}

egl::ConfigSet DisplayVkSimple::generateConfigs()
{
    constexpr GLenum kColorFormats[] = {GL_RGBA8, GL_BGRA8_EXT, GL_RGB565, GL_RGB8};

    return egl_vk::GenerateConfigs(kColorFormats, egl_vk::kConfigDepthStencilFormats, this);
}

// TODO: anglebug.com/40096731
// Detemine if check is needed.
void DisplayVkSimple::checkConfigSupport(egl::Config *config) {}

const char *DisplayVkSimple::getWSIExtension() const
{
    return VK_KHR_DISPLAY_EXTENSION_NAME;
}

bool IsVulkanSimpleDisplayAvailable()
{
    return true;
}

DisplayImpl *CreateVulkanSimpleDisplay(const egl::DisplayState &state)
{
    return new DisplayVkSimple(state);
}

}  // namespace rx
