/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 30, 2022.
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

// ImageEGL.h: Defines the rx::ImageEGL class, the EGL implementation of EGL images

#ifndef LIBANGLE_RENDERER_GL_EGL_IMAGEEGL_H_
#define LIBANGLE_RENDERER_GL_EGL_IMAGEEGL_H_

#include "libANGLE/renderer/gl/ImageGL.h"

namespace egl
{
class AttributeMap;
}

namespace rx
{

class FunctionsEGL;

class ImageEGL final : public ImageGL
{
  public:
    ImageEGL(const egl::ImageState &state,
             const gl::Context *context,
             EGLenum target,
             const egl::AttributeMap &attribs,
             const FunctionsEGL *egl);
    ~ImageEGL() override;

    egl::Error initialize(const egl::Display *display) override;

    angle::Result orphan(const gl::Context *context, egl::ImageSibling *sibling) override;

    angle::Result setTexture2D(const gl::Context *context,
                               gl::TextureType type,
                               TextureGL *texture,
                               GLenum *outInternalFormat) override;
    angle::Result setRenderbufferStorage(const gl::Context *context,
                                         RenderbufferGL *renderbuffer,
                                         GLenum *outInternalFormat) override;

  private:
    const FunctionsEGL *mEGL;

    // State needed for initialization
    EGLContext mContext;
    EGLenum mTarget;
    bool mPreserveImage;

    GLenum mNativeInternalFormat;

    EGLImage mImage;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_EGL_IMAGEEGL_H_
