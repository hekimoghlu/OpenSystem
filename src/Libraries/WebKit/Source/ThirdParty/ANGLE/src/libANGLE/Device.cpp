/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
// Copyright 2015 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Device.cpp: Implements the egl::Device class, representing the abstract
// device. Implements EGLDevice.

#include "libANGLE/Device.h"

#include <iterator>

#include <EGL/eglext.h>
#include <platform/PlatformMethods.h>

#include "anglebase/no_destructor.h"
#include "common/debug.h"
#include "common/platform.h"
#include "libANGLE/renderer/DeviceImpl.h"

#if defined(ANGLE_ENABLE_D3D11)
#    include "libANGLE/renderer/d3d/d3d11/Device11.h"
#endif

namespace egl
{

template <typename T>
static std::string GenerateExtensionsString(const T &extensions)
{
    std::vector<std::string> extensionsVector = extensions.getStrings();

    std::ostringstream stream;
    std::copy(extensionsVector.begin(), extensionsVector.end(),
              std::ostream_iterator<std::string>(stream, " "));
    return stream.str();
}

typedef std::set<egl::Device *> DeviceSet;
static DeviceSet *GetDeviceSet()
{
    static angle::base::NoDestructor<DeviceSet> devices;
    return devices.get();
}

// Static factory methods
egl::Error Device::CreateDevice(EGLint deviceType, void *nativeDevice, Device **outDevice)
{
    *outDevice = nullptr;

    std::unique_ptr<rx::DeviceImpl> newDeviceImpl;

#if defined(ANGLE_ENABLE_D3D11)
    if (deviceType == EGL_D3D11_DEVICE_ANGLE)
    {
        newDeviceImpl.reset(new rx::Device11(nativeDevice));
    }
#endif

    // Note that creating an EGL device from inputted D3D9 parameters isn't currently supported

    if (newDeviceImpl == nullptr)
    {
        return EglBadAttribute();
    }

    ANGLE_TRY(newDeviceImpl->initialize());
    *outDevice = new Device(nullptr, newDeviceImpl.release());

    return NoError();
}

bool Device::IsValidDevice(const Device *device)
{
    const DeviceSet *deviceSet = GetDeviceSet();
    return deviceSet->find(const_cast<Device *>(device)) != deviceSet->end();
}

Device::Device(Display *owningDisplay, rx::DeviceImpl *impl)
    : mLabel(nullptr), mOwningDisplay(owningDisplay), mImplementation(impl)
{
    ASSERT(GetDeviceSet()->find(this) == GetDeviceSet()->end());
    GetDeviceSet()->insert(this);
    initDeviceExtensions();
}

Device::~Device()
{
    ASSERT(GetDeviceSet()->find(this) != GetDeviceSet()->end());
    GetDeviceSet()->erase(this);
}

void Device::setLabel(EGLLabelKHR label)
{
    mLabel = label;
}

EGLLabelKHR Device::getLabel() const
{
    return mLabel;
}

Error Device::getAttribute(EGLint attribute, EGLAttrib *value)
{
    void *nativeAttribute = nullptr;
    egl::Error error =
        getImplementation()->getAttribute(getOwningDisplay(), attribute, &nativeAttribute);
    *value = reinterpret_cast<EGLAttrib>(nativeAttribute);
    return error;
}

void Device::initDeviceExtensions()
{
    mImplementation->generateExtensions(&mDeviceExtensions);
    mDeviceExtensionString = GenerateExtensionsString(mDeviceExtensions);
}

const DeviceExtensions &Device::getExtensions() const
{
    return mDeviceExtensions;
}

const std::string &Device::getExtensionString() const
{
    return mDeviceExtensionString;
}

const std::string &Device::getDeviceString(EGLint name)
{
    if (mDeviceStrings.find(name) == mDeviceStrings.end())
    {
        mDeviceStrings.emplace(name, mImplementation.get()->getDeviceString(name));
    }

    return mDeviceStrings.find(name)->second;
}
}  // namespace egl
