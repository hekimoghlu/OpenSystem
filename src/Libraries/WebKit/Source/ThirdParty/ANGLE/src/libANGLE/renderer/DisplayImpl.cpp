/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 13, 2024.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DisplayImpl.cpp: Implementation methods of egl::Display

#include "libANGLE/renderer/DisplayImpl.h"

#include "libANGLE/Display.h"
#include "libANGLE/Surface.h"
#include "libANGLE/renderer/DeviceImpl.h"

namespace rx
{
namespace
{
// For back-ends that do not implement EGLDevice.
class MockDevice : public DeviceImpl
{
  public:
    MockDevice() = default;
    egl::Error initialize() override { return egl::NoError(); }
    egl::Error getAttribute(const egl::Display *display, EGLint attribute, void **outValue) override
    {
        UNREACHABLE();
        return egl::EglBadAttribute();
    }
    void generateExtensions(egl::DeviceExtensions *outExtensions) const override
    {
        *outExtensions = egl::DeviceExtensions();
    }
};
}  // anonymous namespace

DisplayImpl::DisplayImpl(const egl::DisplayState &state)
    : mState(state), mExtensionsInitialized(false), mCapsInitialized(false), mBlobCache(nullptr)
{}

DisplayImpl::~DisplayImpl()
{
    ASSERT(mState.surfaceMap.empty());
}

egl::Error DisplayImpl::prepareForCall()
{
    return egl::NoError();
}

egl::Error DisplayImpl::releaseThread()
{
    return egl::NoError();
}

const egl::DisplayExtensions &DisplayImpl::getExtensions() const
{
    if (!mExtensionsInitialized)
    {
        generateExtensions(&mExtensions);
        mExtensionsInitialized = true;
    }

    return mExtensions;
}

egl::Error DisplayImpl::handleGPUSwitch()
{
    return egl::NoError();
}

egl::Error DisplayImpl::forceGPUSwitch(EGLint gpuIDHigh, EGLint gpuIDLow)
{
    return egl::NoError();
}

egl::Error DisplayImpl::waitUntilWorkScheduled()
{
    return egl::NoError();
}

egl::Error DisplayImpl::validateClientBuffer(const egl::Config *configuration,
                                             EGLenum buftype,
                                             EGLClientBuffer clientBuffer,
                                             const egl::AttributeMap &attribs) const
{
    UNREACHABLE();
    return egl::EglBadDisplay() << "DisplayImpl::validateClientBuffer unimplemented.";
}

egl::Error DisplayImpl::validateImageClientBuffer(const gl::Context *context,
                                                  EGLenum target,
                                                  EGLClientBuffer clientBuffer,
                                                  const egl::AttributeMap &attribs) const
{
    UNREACHABLE();
    return egl::EglBadDisplay() << "DisplayImpl::validateImageClientBuffer unimplemented.";
}

egl::Error DisplayImpl::validatePixmap(const egl::Config *config,
                                       EGLNativePixmapType pixmap,
                                       const egl::AttributeMap &attributes) const
{
    UNREACHABLE();
    return egl::EglBadDisplay() << "DisplayImpl::valdiatePixmap unimplemented.";
}

const egl::Caps &DisplayImpl::getCaps() const
{
    if (!mCapsInitialized)
    {
        generateCaps(&mCaps);
        mCapsInitialized = true;
    }

    return mCaps;
}

DeviceImpl *DisplayImpl::createDevice()
{
    return new MockDevice();
}

angle::NativeWindowSystem DisplayImpl::getWindowSystem() const
{
    return angle::NativeWindowSystem::Other;
}

bool DisplayImpl::supportsDmaBufFormat(EGLint format) const
{
    UNREACHABLE();
    return false;
}

egl::Error DisplayImpl::queryDmaBufFormats(EGLint max_formats, EGLint *formats, EGLint *num_formats)
{
    UNREACHABLE();
    return egl::NoError();
}

egl::Error DisplayImpl::queryDmaBufModifiers(EGLint format,
                                             EGLint max_modifiers,
                                             EGLuint64KHR *modifiers,
                                             EGLBoolean *external_only,
                                             EGLint *num_modifiers)
{
    UNREACHABLE();
    return egl::NoError();
}

egl::Error DisplayImpl::querySupportedCompressionRates(const egl::Config *configuration,
                                                       const egl::AttributeMap &attributes,
                                                       EGLint *rates,
                                                       EGLint rate_size,
                                                       EGLint *num_rates) const
{
    UNREACHABLE();
    return egl::NoError();
}
}  // namespace rx
