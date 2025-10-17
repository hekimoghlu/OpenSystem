/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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

// NativeBufferImageSiblingAndroid.h: Defines the NativeBufferImageSiblingAndroid to wrap EGL images
// created from ANativeWindowBuffer objects

#ifndef LIBANGLE_RENDERER_GL_EGL_ANDROID_NATIVEBUFFERIMAGESIBLINGANDROID_H_
#define LIBANGLE_RENDERER_GL_EGL_ANDROID_NATIVEBUFFERIMAGESIBLINGANDROID_H_

#include "libANGLE/renderer/gl/egl/ExternalImageSiblingEGL.h"

namespace rx
{

class NativeBufferImageSiblingAndroid : public ExternalImageSiblingEGL
{
  public:
    NativeBufferImageSiblingAndroid(EGLClientBuffer buffer, const egl::AttributeMap &attribs);
    ~NativeBufferImageSiblingAndroid() override;

    egl::Error initialize(const egl::Display *display) override;

    // ExternalImageSiblingImpl interface
    gl::Format getFormat() const override;
    bool isRenderable(const gl::Context *context) const override;
    bool isTexturable(const gl::Context *context) const override;
    bool isYUV() const override;
    bool hasProtectedContent() const override;
    gl::Extents getSize() const override;
    size_t getSamples() const override;

    // ExternalImageSiblingEGL interface
    EGLClientBuffer getBuffer() const override;
    void getImageCreationAttributes(std::vector<EGLint> *outAttributes) const override;

  private:
    EGLClientBuffer mBuffer;
    gl::Extents mSize;
    gl::Format mFormat;
    bool mYUV;
    bool mHasProtectedContent;
    GLint mColorSpace;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_ANDROID_NATIVEBUFFERIMAGESIBLINGANDROID_H_
