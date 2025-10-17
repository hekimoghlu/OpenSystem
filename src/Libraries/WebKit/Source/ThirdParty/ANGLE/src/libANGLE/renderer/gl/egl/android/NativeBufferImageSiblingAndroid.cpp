/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 10, 2024.
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
// Copyright 2018 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// NativeBufferImageSiblingAndroid.cpp: Implements the NativeBufferImageSiblingAndroid class

#include "libANGLE/renderer/gl/egl/android/NativeBufferImageSiblingAndroid.h"

#include "common/android_util.h"

namespace rx
{
NativeBufferImageSiblingAndroid::NativeBufferImageSiblingAndroid(EGLClientBuffer buffer,
                                                                 const egl::AttributeMap &attribs)
    : mBuffer(buffer), mFormat(GL_NONE), mYUV(false), mColorSpace(EGL_GL_COLORSPACE_LINEAR_KHR)
{
    if (attribs.contains(EGL_GL_COLORSPACE_KHR))
    {
        mColorSpace = attribs.getAsInt(EGL_GL_COLORSPACE_KHR);
    }
}

NativeBufferImageSiblingAndroid::~NativeBufferImageSiblingAndroid() {}

egl::Error NativeBufferImageSiblingAndroid::initialize(const egl::Display *display)
{
    int pixelFormat = 0;
    uint64_t usage  = 0;
    angle::android::GetANativeWindowBufferProperties(
        angle::android::ClientBufferToANativeWindowBuffer(mBuffer), &mSize.width, &mSize.height,
        &mSize.depth, &pixelFormat, &usage);
    mFormat = gl::Format(angle::android::NativePixelFormatToGLInternalFormat(pixelFormat));
    mYUV    = angle::android::NativePixelFormatIsYUV(pixelFormat);
    mHasProtectedContent = false;

    return egl::NoError();
}

gl::Format NativeBufferImageSiblingAndroid::getFormat() const
{
    return mFormat;
}

bool NativeBufferImageSiblingAndroid::isRenderable(const gl::Context *context) const
{
    return true;
}

bool NativeBufferImageSiblingAndroid::isTexturable(const gl::Context *context) const
{
    return true;
}

bool NativeBufferImageSiblingAndroid::isYUV() const
{
    return mYUV;
}

bool NativeBufferImageSiblingAndroid::hasProtectedContent() const
{
    return mHasProtectedContent;
}

gl::Extents NativeBufferImageSiblingAndroid::getSize() const
{
    return mSize;
}

size_t NativeBufferImageSiblingAndroid::getSamples() const
{
    return 0;
}

EGLClientBuffer NativeBufferImageSiblingAndroid::getBuffer() const
{
    return mBuffer;
}

void NativeBufferImageSiblingAndroid::getImageCreationAttributes(
    std::vector<EGLint> *outAttributes) const
{
    if (mColorSpace != EGL_GL_COLORSPACE_LINEAR_KHR)
    {
        outAttributes->push_back(EGL_GL_COLORSPACE_KHR);
        outAttributes->push_back(mColorSpace);
    }
}

}  // namespace rx
