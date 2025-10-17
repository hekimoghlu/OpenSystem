/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 8, 2022.
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

// RenderbufferGL.h: Defines the class interface for RenderbufferGL.

#ifndef LIBANGLE_RENDERER_GL_RENDERBUFFERGL_H_
#define LIBANGLE_RENDERER_GL_RENDERBUFFERGL_H_

#include "libANGLE/renderer/RenderbufferImpl.h"

namespace angle
{
struct FeaturesGL;
}  // namespace angle

namespace gl
{
class TextureCapsMap;
}  // namespace gl

namespace rx
{

class BlitGL;
class FunctionsGL;
class StateManagerGL;

class RenderbufferGL : public RenderbufferImpl
{
  public:
    RenderbufferGL(const gl::RenderbufferState &state, GLuint id);
    ~RenderbufferGL() override;

    void onDestroy(const gl::Context *context) override;

    angle::Result setStorage(const gl::Context *context,
                             GLenum internalformat,
                             GLsizei width,
                             GLsizei height) override;
    angle::Result setStorageMultisample(const gl::Context *context,
                                        GLsizei samples,
                                        GLenum internalformat,
                                        GLsizei width,
                                        GLsizei height,
                                        gl::MultisamplingMode mode) override;
    angle::Result setStorageEGLImageTarget(const gl::Context *context, egl::Image *image) override;

    angle::Result initializeContents(const gl::Context *context,
                                     GLenum binding,
                                     const gl::ImageIndex &imageIndex) override;

    GLuint getRenderbufferID() const;
    GLenum getNativeInternalFormat() const;

  private:
    GLuint mRenderbufferID;

    GLenum mNativeInternalFormat;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_GL_RENDERBUFFERGL_H_
