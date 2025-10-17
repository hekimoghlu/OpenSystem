/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 26, 2022.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// DeviceVkLinux.cpp:
//    Implements the class methods for DeviceVkLinux.
//

#include "libANGLE/renderer/vulkan/linux/DeviceVkLinux.h"

#include <unistd.h>

#include "common/debug.h"
#include "common/vulkan/vulkan_icd.h"
#include "libANGLE/Display.h"
#include "libANGLE/renderer/vulkan/DisplayVk.h"
#include "libANGLE/renderer/vulkan/vk_renderer.h"

namespace rx
{

DeviceVkLinux::DeviceVkLinux(DisplayVk *display) : mDisplay(display) {}

egl::Error DeviceVkLinux::initialize()
{
    vk::Renderer *renderer                         = mDisplay->getRenderer();
    VkPhysicalDeviceDrmPropertiesEXT drmProperties = renderer->getPhysicalDeviceDrmProperties();

    // Unfortunately `VkPhysicalDeviceDrmPropertiesEXT` doesn't give us the information about the
    // filesystem layout needed by the EGL version. As ChromeOS/Exo is currently the only user,
    // implement the extension only for Linux where we can reasonably assume `/dev/dri/...` file
    // paths.
    if (drmProperties.hasPrimary)
    {
        char deviceName[50];
        const long long primaryMinor = drmProperties.primaryMinor;
        snprintf(deviceName, sizeof(deviceName), "/dev/dri/card%lld", primaryMinor);

        if (access(deviceName, F_OK) != -1)
        {
            mDrmDevice = deviceName;
        }
    }
    if (drmProperties.hasRender)
    {
        char deviceName[50];
        const long long renderMinor = drmProperties.renderMinor;
        snprintf(deviceName, sizeof(deviceName), "/dev/dri/renderD%lld", renderMinor);

        if (access(deviceName, F_OK) != -1)
        {
            mDrmRenderNode = deviceName;
        }
    }

    if (mDrmDevice.empty() && !mDrmRenderNode.empty())
    {
        mDrmDevice = mDrmRenderNode;
    }

    return egl::NoError();
}

void DeviceVkLinux::generateExtensions(egl::DeviceExtensions *outExtensions) const
{
    DeviceVk::generateExtensions(outExtensions);

    if (!mDrmDevice.empty())
    {
        outExtensions->deviceDrmEXT = true;
    }
    if (!mDrmRenderNode.empty())
    {
        outExtensions->deviceDrmRenderNodeEXT = true;
    }
}

const std::string DeviceVkLinux::getDeviceString(EGLint name)
{
    switch (name)
    {
        case EGL_DRM_DEVICE_FILE_EXT:
            return mDrmDevice;
        case EGL_DRM_RENDER_NODE_FILE_EXT:
            return mDrmRenderNode;
        default:
            UNIMPLEMENTED();
            return std::string();
    }
}

}  // namespace rx
