/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 28, 2023.
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

// Framebuffer9.h: Defines the Framebuffer9 class.

#ifndef LIBANGLE_RENDERER_D3D_D3D9_FRAMBUFFER9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_FRAMBUFFER9_H_

#include "libANGLE/renderer/RenderTargetCache.h"
#include "libANGLE/renderer/d3d/FramebufferD3D.h"
#include "libANGLE/renderer/d3d/d3d9/renderer9_utils.h"

namespace rx
{
class Renderer9;

class Framebuffer9 : public FramebufferD3D
{
  public:
    Framebuffer9(const gl::FramebufferState &data, Renderer9 *renderer);
    ~Framebuffer9() override;

    angle::Result discard(const gl::Context *context,
                          size_t count,
                          const GLenum *attachments) override;
    angle::Result invalidate(const gl::Context *context,
                             size_t count,
                             const GLenum *attachments) override;
    angle::Result invalidateSub(const gl::Context *context,
                                size_t count,
                                const GLenum *attachments,
                                const gl::Rectangle &area) override;

    angle::Result getSamplePosition(const gl::Context *context,
                                    size_t index,
                                    GLfloat *xy) const override;

    angle::Result syncState(const gl::Context *context,
                            GLenum binding,
                            const gl::Framebuffer::DirtyBits &dirtyBits,
                            gl::Command command) override;

    const gl::AttachmentArray<RenderTarget9 *> &getCachedColorRenderTargets() const
    {
        return mRenderTargetCache.getColors();
    }

    const RenderTarget9 *getCachedDepthStencilRenderTarget() const
    {
        return mRenderTargetCache.getDepthStencil();
    }

    const gl::InternalFormat &getImplementationColorReadFormat(
        const gl::Context *context) const override;

  private:
    angle::Result clearImpl(const gl::Context *context,
                            const ClearParameters &clearParams) override;

    angle::Result readPixelsImpl(const gl::Context *context,
                                 const gl::Rectangle &area,
                                 GLenum format,
                                 GLenum type,
                                 size_t outputPitch,
                                 const gl::PixelPackState &pack,
                                 gl::Buffer *packPixels,
                                 uint8_t *pixels) override;

    angle::Result blitImpl(const gl::Context *context,
                           const gl::Rectangle &sourceArea,
                           const gl::Rectangle &destArea,
                           const gl::Rectangle *scissor,
                           bool blitRenderTarget,
                           bool blitDepth,
                           bool blitStencil,
                           GLenum filter,
                           const gl::Framebuffer *sourceFramebuffer) override;

    Renderer9 *const mRenderer;

    RenderTargetCache<RenderTarget9> mRenderTargetCache;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_FRAMBUFFER9_H_
