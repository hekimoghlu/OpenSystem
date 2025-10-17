/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
// DisplayVkMac.mm:
//    Implements methods from DisplayVkMac
//

#include "libANGLE/renderer/vulkan/mac/DisplayVkMac.h"

#include <vulkan/vulkan.h>

#include "libANGLE/renderer/vulkan/mac/IOSurfaceSurfaceVkMac.h"
#include "libANGLE/renderer/vulkan/mac/WindowSurfaceVkMac.h"
#include "libANGLE/renderer/vulkan/vk_caps_utils.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

#import <Cocoa/Cocoa.h>

namespace rx
{

DisplayVkMac::DisplayVkMac(const egl::DisplayState &state) : DisplayVk(state) {}

bool DisplayVkMac::isValidNativeWindow(EGLNativeWindowType window) const
{
    NSObject *layer = reinterpret_cast<NSObject *>(window);
    return [layer isKindOfClass:[CALayer class]];
}

SurfaceImpl *DisplayVkMac::createWindowSurfaceVk(const egl::SurfaceState &state,
                                                 EGLNativeWindowType window)
{
    ASSERT(isValidNativeWindow(window));
    return new WindowSurfaceVkMac(state, window);
}

SurfaceImpl *DisplayVkMac::createPbufferFromClientBuffer(const egl::SurfaceState &state,
                                                         EGLenum buftype,
                                                         EGLClientBuffer clientBuffer,
                                                         const egl::AttributeMap &attribs)
{
    ASSERT(buftype == EGL_IOSURFACE_ANGLE);

    return new IOSurfaceSurfaceVkMac(state, clientBuffer, attribs, mRenderer);
}

egl::ConfigSet DisplayVkMac::generateConfigs()
{
    constexpr GLenum kColorFormats[] = {GL_BGRA8_EXT, GL_BGRX8_ANGLEX};
    return egl_vk::GenerateConfigs(kColorFormats, egl_vk::kConfigDepthStencilFormats, this);
}

void DisplayVkMac::checkConfigSupport(egl::Config *config)
{
    // TODO(geofflang): Test for native support and modify the config accordingly.
    // anglebug.com/42261400
}

const char *DisplayVkMac::getWSIExtension() const
{
    return VK_EXT_METAL_SURFACE_EXTENSION_NAME;
}

bool IsVulkanMacDisplayAvailable()
{
    return true;
}

DisplayImpl *CreateVulkanMacDisplay(const egl::DisplayState &state)
{
    return new DisplayVkMac(state);
}

void DisplayVkMac::generateExtensions(egl::DisplayExtensions *outExtensions) const
{
    outExtensions->iosurfaceClientBuffer = true;

    DisplayVk::generateExtensions(outExtensions);
}

egl::Error DisplayVkMac::validateClientBuffer(const egl::Config *configuration,
                                              EGLenum buftype,
                                              EGLClientBuffer clientBuffer,
                                              const egl::AttributeMap &attribs) const
{
    ASSERT(buftype == EGL_IOSURFACE_ANGLE);

    if (!IOSurfaceSurfaceVkMac::ValidateAttributes(this, clientBuffer, attribs))
    {
        return egl::EglBadAttribute();
    }
    return egl::NoError();
}

}  // namespace rx
