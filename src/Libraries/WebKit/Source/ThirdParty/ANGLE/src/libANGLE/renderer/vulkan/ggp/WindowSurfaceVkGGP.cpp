/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
// WindowSurfaceVkGGP.cpp:
//    Implements the class methods for WindowSurfaceVkGGP.
//

#include "libANGLE/renderer/vulkan/ggp/WindowSurfaceVkGGP.h"

#include "libANGLE/Context.h"
#include "libANGLE/Display.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/vulkan/DisplayVk.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{
namespace
{
constexpr EGLAttrib kDefaultStreamDescriptor = static_cast<EGLAttrib>(kGgpPrimaryStreamDescriptor);
}  // namespace

WindowSurfaceVkGGP::WindowSurfaceVkGGP(const egl::SurfaceState &surfaceState,
                                       EGLNativeWindowType window)
    : WindowSurfaceVk(surfaceState, window)
{}

angle::Result WindowSurfaceVkGGP::createSurfaceVk(vk::ErrorContext *context,
                                                  gl::Extents *extentsOut)
{
    vk::Renderer *renderer = context->getRenderer();

    // Get the stream descriptor if specified. Default is kGgpPrimaryStreamDescriptor.
    EGLAttrib streamDescriptor =
        mState.attributes.get(EGL_GGP_STREAM_DESCRIPTOR_ANGLE, kDefaultStreamDescriptor);

    VkStreamDescriptorSurfaceCreateInfoGGP createInfo = {};
    createInfo.sType            = VK_STRUCTURE_TYPE_STREAM_DESCRIPTOR_SURFACE_CREATE_INFO_GGP;
    createInfo.streamDescriptor = static_cast<GgpStreamDescriptor>(streamDescriptor);

    ANGLE_VK_TRY(context, vkCreateStreamDescriptorSurfaceGGP(renderer->getInstance(), &createInfo,
                                                             nullptr, &mSurface));

    return getCurrentWindowSize(context, extentsOut);
}

angle::Result WindowSurfaceVkGGP::getCurrentWindowSize(vk::ErrorContext *context,
                                                       gl::Extents *extentsOut)
{
    vk::Renderer *renderer                 = context->getRenderer();
    const VkPhysicalDevice &physicalDevice = renderer->getPhysicalDevice();

    ANGLE_VK_TRY(context, vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, mSurface,
                                                                    &mSurfaceCaps));

    *extentsOut =
        gl::Extents(mSurfaceCaps.currentExtent.width, mSurfaceCaps.currentExtent.height, 1);
    return angle::Result::Continue;
}

egl::Error WindowSurfaceVkGGP::swapWithFrameToken(const gl::Context *context,
                                                  EGLFrameTokenANGLE frameToken)
{
    VkPresentFrameTokenGGP frameTokenData = {};
    frameTokenData.sType                  = VK_STRUCTURE_TYPE_PRESENT_FRAME_TOKEN_GGP;
    frameTokenData.frameToken             = static_cast<GgpFrameToken>(frameToken);

    angle::Result result = swapImpl(context, nullptr, 0, &frameTokenData);
    return angle::ToEGL(result, EGL_BAD_SURFACE);
}
}  // namespace rx
