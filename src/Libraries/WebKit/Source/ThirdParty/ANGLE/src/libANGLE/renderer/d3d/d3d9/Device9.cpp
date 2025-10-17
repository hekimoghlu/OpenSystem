/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

// Device9.cpp: D3D9 implementation of egl::Device

#include "libANGLE/renderer/d3d/d3d9/Device9.h"

#include "libANGLE/Device.h"
#include "libANGLE/Display.h"

#include <EGL/eglext.h>

namespace rx
{

Device9::Device9(IDirect3DDevice9 *device) : mDevice(device) {}

Device9::~Device9() {}

egl::Error Device9::getAttribute(const egl::Display *display, EGLint attribute, void **outValue)
{
    ASSERT(attribute == EGL_D3D9_DEVICE_ANGLE);
    *outValue = mDevice;
    return egl::NoError();
}

egl::Error Device9::initialize()
{
    return egl::NoError();
}

void Device9::generateExtensions(egl::DeviceExtensions *outExtensions) const
{
    outExtensions->deviceD3D  = true;
    outExtensions->deviceD3D9 = true;
}

}  // namespace rx
