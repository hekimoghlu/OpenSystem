/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 21, 2025.
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
// Copyright 2022 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DisplayVkGbm.cpp:
//    Implements the class methods for DisplayVkGbm.
//

#include "libANGLE/renderer/vulkan/linux/gbm/DisplayVkGbm.h"

#include <gbm.h>

#include "common/linux/dma_buf_utils.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/vulkan/vk_caps_utils.h"

namespace rx
{

DisplayVkGbm::DisplayVkGbm(const egl::DisplayState &state)
    : DisplayVkLinux(state), mGbmDevice(nullptr)
{}

egl::Error DisplayVkGbm::initialize(egl::Display *display)
{
    mGbmDevice = reinterpret_cast<gbm_device *>(display->getNativeDisplayId());
    if (!mGbmDevice)
    {
        ERR() << "Failed to retrieve GBM device";
        return egl::EglNotInitialized();
    }

    return DisplayVk::initialize(display);
}

void DisplayVkGbm::terminate()
{
    mGbmDevice = nullptr;
    DisplayVk::terminate();
}

bool DisplayVkGbm::isValidNativeWindow(EGLNativeWindowType window) const
{
    return (void *)window != nullptr;
}

SurfaceImpl *DisplayVkGbm::createWindowSurfaceVk(const egl::SurfaceState &state,
                                                 EGLNativeWindowType window)
{
    return nullptr;
}

egl::ConfigSet DisplayVkGbm::generateConfigs()
{
    const std::array<GLenum, 1> kColorFormats = {GL_BGRA8_EXT};

    std::vector<GLenum> depthStencilFormats(
        egl_vk::kConfigDepthStencilFormats,
        egl_vk::kConfigDepthStencilFormats + ArraySize(egl_vk::kConfigDepthStencilFormats));

    if (getCaps().stencil8)
    {
        depthStencilFormats.push_back(GL_STENCIL_INDEX8);
    }

    egl::ConfigSet cfgSet =
        egl_vk::GenerateConfigs(kColorFormats.data(), kColorFormats.size(),
                                depthStencilFormats.data(), depthStencilFormats.size(), this);

    return cfgSet;
}

void DisplayVkGbm::checkConfigSupport(egl::Config *config) {}

const char *DisplayVkGbm::getWSIExtension() const
{
    return nullptr;
}

angle::NativeWindowSystem DisplayVkGbm::getWindowSystem() const
{
    return angle::NativeWindowSystem::Gbm;
}

bool IsVulkanGbmDisplayAvailable()
{
    return true;
}

DisplayImpl *CreateVulkanGbmDisplay(const egl::DisplayState &state)
{
    return new DisplayVkGbm(state);
}

}  // namespace rx
