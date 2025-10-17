/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

// Device11.cpp: D3D11 implementation of egl::Device

#include "libANGLE/renderer/d3d/d3d11/Device11.h"

#include "libANGLE/Device.h"
#include "libANGLE/Display.h"

#include <EGL/eglext.h>

namespace rx
{

Device11::Device11(void *nativeDevice)
{
    // Validate the device
    IUnknown *iunknown = static_cast<IUnknown *>(nativeDevice);

    // The QI to ID3D11Device adds a ref to the D3D11 device.
    // Deliberately don't release the ref here, so that the Device11 holds a ref to the
    // D3D11 device.
    iunknown->QueryInterface(__uuidof(ID3D11Device), reinterpret_cast<void **>(&mDevice));
}

Device11::~Device11()
{
    if (mDevice)
    {
        // Device11 holds a ref to an externally-sourced D3D11 device. We must release it.
        mDevice->Release();
        mDevice = nullptr;
    }
}

egl::Error Device11::getAttribute(const egl::Display *display, EGLint attribute, void **outValue)
{
    ASSERT(attribute == EGL_D3D11_DEVICE_ANGLE);
    *outValue = mDevice;
    return egl::NoError();
}

egl::Error Device11::initialize()
{
    if (!mDevice)
    {
        return egl::EglBadAttribute() << "Invalid D3D device passed into EGLDeviceEXT";
    }

    return egl::NoError();
}

void Device11::generateExtensions(egl::DeviceExtensions *outExtensions) const
{
    outExtensions->deviceD3D   = true;
    outExtensions->deviceD3D11 = true;
}

}  // namespace rx
