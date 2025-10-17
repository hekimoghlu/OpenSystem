/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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

// FramebufferImpl.h: Defines the abstract rx::FramebufferImpl class.

#ifndef LIBANGLE_RENDERER_FRAMEBUFFERIMPL_H_
#define LIBANGLE_RENDERER_FRAMEBUFFERIMPL_H_

#include "angle_gl.h"
#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/State.h"
#include "libANGLE/angletypes.h"

namespace gl
{
class Buffer;
class Framebuffer;
class FramebufferAttachment;
struct PixelPackState;
}  // namespace gl

namespace rx
{
class DisplayImpl;

class FramebufferImpl : angle::NonCopyable
{
  public:
    explicit FramebufferImpl(const gl::FramebufferState &state) : mState(state) {}
    virtual ~FramebufferImpl() {}
    virtual void destroy(const gl::Context *context) {}

    virtual angle::Result discard(const gl::Context *context,
                                  size_t count,
                                  const GLenum *attachments)       = 0;
    virtual angle::Result invalidate(const gl::Context *context,
                                     size_t count,
                                     const GLenum *attachments)    = 0;
    virtual angle::Result invalidateSub(const gl::Context *context,
                                        size_t count,
                                        const GLenum *attachments,
                                        const gl::Rectangle &area) = 0;

    virtual angle::Result clear(const gl::Context *context, GLbitfield mask) = 0;
    virtual angle::Result clearBufferfv(const gl::Context *context,
                                        GLenum buffer,
                                        GLint drawbuffer,
                                        const GLfloat *values)               = 0;
    virtual angle::Result clearBufferuiv(const gl::Context *context,
                                         GLenum buffer,
                                         GLint drawbuffer,
                                         const GLuint *values)               = 0;
    virtual angle::Result clearBufferiv(const gl::Context *context,
                                        GLenum buffer,
                                        GLint drawbuffer,
                                        const GLint *values)                 = 0;
    virtual angle::Result clearBufferfi(const gl::Context *context,
                                        GLenum buffer,
                                        GLint drawbuffer,
                                        GLfloat depth,
                                        GLint stencil)                       = 0;

    virtual const gl::InternalFormat &getImplementationColorReadFormat(
        const gl::Context *context) const;
    virtual angle::Result readPixels(const gl::Context *context,
                                     const gl::Rectangle &area,
                                     GLenum format,
                                     GLenum type,
                                     const gl::PixelPackState &pack,
                                     gl::Buffer *packBuffer,
                                     void *pixels) = 0;

    virtual angle::Result blit(const gl::Context *context,
                               const gl::Rectangle &sourceArea,
                               const gl::Rectangle &destArea,
                               GLbitfield mask,
                               GLenum filter) = 0;

    virtual gl::FramebufferStatus checkStatus(const gl::Context *context) const = 0;

    virtual angle::Result ensureAttachmentsInitialized(const gl::Context *context,
                                                       const gl::DrawBufferMask &colorAttachments,
                                                       bool depth,
                                                       bool stencil);

    virtual angle::Result syncState(const gl::Context *context,
                                    GLenum binding,
                                    const gl::Framebuffer::DirtyBits &dirtyBits,
                                    gl::Command command) = 0;

    virtual angle::Result getSamplePosition(const gl::Context *context,
                                            size_t index,
                                            GLfloat *xy) const = 0;

    // Special configuration option for checkStatus(). Some back-ends don't require a syncState
    // before calling checkStatus. In practice the GL back-end is the only config that needs
    // syncState because it depends on the behaviour of the driver. Allowing the Vulkan and
    // D3D back-ends to skip syncState lets us do more work in the syncState call.
    virtual bool shouldSyncStateBeforeCheckStatus() const;

    virtual angle::Result onLabelUpdate(const gl::Context *context);

    const gl::FramebufferState &getState() const { return mState; }

  protected:
    const gl::FramebufferState &mState;
};

inline bool FramebufferImpl::shouldSyncStateBeforeCheckStatus() const
{
    return false;
}

// Default implementation returns the format specified in the attachment.
inline const gl::InternalFormat &FramebufferImpl::getImplementationColorReadFormat(
    const gl::Context *context) const
{
    const gl::FramebufferAttachment *readAttachment = mState.getReadAttachment();
    return *readAttachment->getFormat().info;
}
}  // namespace rx

#endif  // LIBANGLE_RENDERER_FRAMEBUFFERIMPL_H_
