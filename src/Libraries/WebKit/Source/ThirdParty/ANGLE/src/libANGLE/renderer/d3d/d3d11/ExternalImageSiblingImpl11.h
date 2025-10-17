/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 2, 2024.
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
// Copyright 2019 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

#ifndef LIBANGLE_RENDERER_D3D_D3D11_EXTERNALIMAGESIBLINGIMPL11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_EXTERNALIMAGESIBLINGIMPL11_H_

#include "libANGLE/renderer/ImageImpl.h"
#include "libANGLE/renderer/d3d/d3d11/renderer11_utils.h"

namespace rx
{

class Renderer11;
class RenderTargetD3D;

class ExternalImageSiblingImpl11 : public ExternalImageSiblingImpl
{
  public:
    ExternalImageSiblingImpl11(Renderer11 *renderer,
                               EGLClientBuffer clientBuffer,
                               const egl::AttributeMap &attribs);
    ~ExternalImageSiblingImpl11() override;

    // ExternalImageSiblingImpl interface
    egl::Error initialize(const egl::Display *display) override;
    gl::Format getFormat() const override;
    bool isRenderable(const gl::Context *context) const override;
    bool isTexturable(const gl::Context *context) const override;
    bool isYUV() const override;
    bool hasProtectedContent() const override;
    gl::Extents getSize() const override;
    size_t getSamples() const override;

    // FramebufferAttachmentObjectImpl interface
    angle::Result getAttachmentRenderTarget(const gl::Context *context,
                                            GLenum binding,
                                            const gl::ImageIndex &imageIndex,
                                            GLsizei samples,
                                            FramebufferAttachmentRenderTarget **rtOut) override;
    angle::Result initializeContents(const gl::Context *context,
                                     GLenum binding,
                                     const gl::ImageIndex &imageIndex) override;

  private:
    angle::Result createRenderTarget(const gl::Context *context);

    Renderer11 *mRenderer;
    EGLClientBuffer mBuffer;
    egl::AttributeMap mAttribs;

    TextureHelper11 mTexture;

    gl::Format mFormat   = gl::Format::Invalid();
    bool mIsRenderable   = false;
    bool mIsTexturable   = false;
    bool mIsTextureArray = false;
    EGLint mWidth        = 0;
    EGLint mHeight       = 0;
    GLsizei mSamples     = 0;
    UINT mArraySlice     = 0;

    std::unique_ptr<RenderTargetD3D> mRenderTarget;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_EXTERNALIMAGESIBLINGIMPL11_H_
