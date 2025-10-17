/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 21, 2023.
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
// FramebufferWgpu.h:
//    Defines the class interface for FramebufferWgpu, implementing FramebufferImpl.
//

#ifndef LIBANGLE_RENDERER_WGPU_FRAMEBUFFERWGPU_H_
#define LIBANGLE_RENDERER_WGPU_FRAMEBUFFERWGPU_H_

#include "libANGLE/renderer/FramebufferImpl.h"
#include "libANGLE/renderer/RenderTargetCache.h"
#include "libANGLE/renderer/wgpu/RenderTargetWgpu.h"

namespace rx
{

class FramebufferWgpu : public FramebufferImpl
{
  public:
    FramebufferWgpu(const gl::FramebufferState &state);
    ~FramebufferWgpu() override;

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

    angle::Result clear(const gl::Context *context, GLbitfield mask) override;
    angle::Result clearBufferfv(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLfloat *values) override;
    angle::Result clearBufferuiv(const gl::Context *context,
                                 GLenum buffer,
                                 GLint drawbuffer,
                                 const GLuint *values) override;
    angle::Result clearBufferiv(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                const GLint *values) override;
    angle::Result clearBufferfi(const gl::Context *context,
                                GLenum buffer,
                                GLint drawbuffer,
                                GLfloat depth,
                                GLint stencil) override;

    angle::Result readPixels(const gl::Context *context,
                             const gl::Rectangle &area,
                             GLenum format,
                             GLenum type,
                             const gl::PixelPackState &pack,
                             gl::Buffer *packBuffer,
                             void *pixels) override;

    angle::Result blit(const gl::Context *context,
                       const gl::Rectangle &sourceArea,
                       const gl::Rectangle &destArea,
                       GLbitfield mask,
                       GLenum filter) override;

    gl::FramebufferStatus checkStatus(const gl::Context *context) const override;

    angle::Result syncState(const gl::Context *context,
                            GLenum binding,
                            const gl::Framebuffer::DirtyBits &dirtyBits,
                            gl::Command command) override;

    angle::Result getSamplePosition(const gl::Context *context,
                                    size_t index,
                                    GLfloat *xy) const override;

    RenderTargetWgpu *getReadPixelsRenderTarget() const;

    void addNewColorAttachments(std::vector<wgpu::RenderPassColorAttachment> newColorAttachments);

    void updateDepthStencilAttachment(
        wgpu::RenderPassDepthStencilAttachment newRenderPassDepthStencilAttachment);

    angle::Result flushOneColorAttachmentUpdate(const gl::Context *context,
                                                bool deferClears,
                                                uint32_t colorIndexGL);

    angle::Result flushAttachmentUpdates(const gl::Context *context,
                                         gl::DrawBufferMask dirtyColorAttachments,
                                         bool dirtyDepthStencilAttachment,
                                         bool deferColorClears,
                                         bool deferDepthStencilClears);

    angle::Result flushDeferredClears(ContextWgpu *contextWgpu);

    // Starts a new render pass iff there are new color and/or depth/stencil attachments.
    angle::Result startRenderPassNewAttachments(ContextWgpu *contextWgpu);

    angle::Result startNewRenderPass(ContextWgpu *contextWgpu);

    void setUpForRenderPass(ContextWgpu *contextWgpu,
                            bool depthOrStencil,
                            std::vector<wgpu::RenderPassColorAttachment> colorAttachments,
                            wgpu::RenderPassDepthStencilAttachment depthStencilAttachment);

    const gl::DrawBuffersArray<wgpu::TextureFormat> &getCurrentColorAttachmentFormats() const
    {
        return mCurrentColorAttachmentFormats;
    }

    wgpu::TextureFormat getCurrentDepthStencilAttachmentFormat() const
    {
        return mCurrentDepthStencilFormat;
    }

  private:
    void mergeClearWithDeferredClears(wgpu::Color clearValue,
                                      gl::DrawBufferMask clearColorBuffers,
                                      float depthValue,
                                      uint32_t stencilValue,
                                      bool clearColor,
                                      bool clearDepth,
                                      bool clearStencil);

    RenderTargetCache<RenderTargetWgpu> mRenderTargetCache;
    wgpu::RenderPassDescriptor mCurrentRenderPassDesc;
    wgpu::RenderPassDepthStencilAttachment mCurrentDepthStencilAttachment;
    std::vector<wgpu::RenderPassColorAttachment> mCurrentColorAttachments;
    gl::DrawBuffersArray<wgpu::TextureFormat> mCurrentColorAttachmentFormats;
    wgpu::TextureFormat mCurrentDepthStencilFormat = wgpu::TextureFormat::Undefined;

    // Secondary vector to track new clears that are added and should be run in a new render pass
    // during flushColorAttachmentUpdates.
    std::vector<wgpu::RenderPassColorAttachment> mNewColorAttachments;
    wgpu::RenderPassDepthStencilAttachment mNewDepthStencilAttachment;
    bool mAddedNewDepthStencilAttachment = false;

    webgpu::ClearValuesArray mDeferredClears;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_WGPU_FRAMEBUFFERWGPU_H_
