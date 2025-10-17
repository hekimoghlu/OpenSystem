/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 22, 2024.
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
// Copyright The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// DmaBufImageSiblingEGL.h: Defines the DmaBufImageSiblingEGL to wrap EGL images
// created from dma_buf objects

#ifndef LIBANGLE_RENDERER_GL_EGL_DMABUFIMAGESIBLINGEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_DMABUFIMAGESIBLINGEGL_H_

#include "libANGLE/renderer/gl/egl/ExternalImageSiblingEGL.h"

namespace rx
{

class DmaBufImageSiblingEGL : public ExternalImageSiblingEGL
{
  public:
    DmaBufImageSiblingEGL(const egl::AttributeMap &attribs);
    ~DmaBufImageSiblingEGL() override;

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
    egl::AttributeMap mAttribs;
    gl::Extents mSize;
    gl::Format mFormat;
    bool mYUV;
    bool mHasProtectedContent;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_DMABUFIMAGESIBLINGEGL_H_
