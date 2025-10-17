/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

// TextureImpl_mock.h: Defines a mock of the TextureImpl class.

#ifndef LIBANGLE_RENDERER_TEXTUREIMPLMOCK_H_
#define LIBANGLE_RENDERER_TEXTUREIMPLMOCK_H_

#include "gmock/gmock.h"

#include "libANGLE/renderer/TextureImpl.h"

namespace rx
{

class MockTextureImpl : public TextureImpl
{
  public:
    MockTextureImpl() : TextureImpl(mMockState), mMockState(gl::TextureType::_2D) {}
    virtual ~MockTextureImpl() { destructor(); }
    MOCK_METHOD9(setImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               GLenum,
                               const gl::Extents &,
                               GLenum,
                               GLenum,
                               const gl::PixelUnpackState &,
                               gl::Buffer *,
                               const uint8_t *));
    MOCK_METHOD8(setSubImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               const gl::Box &,
                               GLenum,
                               GLenum,
                               const gl::PixelUnpackState &,
                               gl::Buffer *,
                               const uint8_t *));
    MOCK_METHOD7(setCompressedImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               GLenum,
                               const gl::Extents &,
                               const gl::PixelUnpackState &,
                               size_t,
                               const uint8_t *));
    MOCK_METHOD7(setCompressedSubImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               const gl::Box &,
                               GLenum,
                               const gl::PixelUnpackState &,
                               size_t,
                               const uint8_t *));
    MOCK_METHOD5(copyImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               const gl::Rectangle &,
                               GLenum,
                               gl::Framebuffer *));
    MOCK_METHOD5(copySubImage,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               const gl::Offset &,
                               const gl::Rectangle &,
                               gl::Framebuffer *));
    MOCK_METHOD9(copyTexture,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               GLenum,
                               GLenum,
                               GLint,
                               bool,
                               bool,
                               bool,
                               const gl::Texture *));
    MOCK_METHOD9(copySubTexture,
                 angle::Result(const gl::Context *,
                               const gl::ImageIndex &,
                               const gl::Offset &,
                               GLint,
                               const gl::Box &,
                               bool,
                               bool,
                               bool,
                               const gl::Texture *));
    MOCK_METHOD2(copyCompressedTexture,
                 angle::Result(const gl::Context *, const gl::Texture *source));
    MOCK_METHOD5(
        setStorage,
        angle::Result(const gl::Context *, gl::TextureType, size_t, GLenum, const gl::Extents &));
    MOCK_METHOD10(setStorageExternalMemory,
                  angle::Result(const gl::Context *,
                                gl::TextureType,
                                size_t,
                                GLenum,
                                const gl::Extents &,
                                gl::MemoryObject *,
                                GLuint64,
                                GLbitfield,
                                GLbitfield,
                                const void *));
    MOCK_METHOD4(setImageExternal,
                 angle::Result(const gl::Context *,
                               gl::TextureType,
                               egl::Stream *,
                               const egl::Stream::GLTextureDescription &));
    MOCK_METHOD3(setEGLImageTarget,
                 angle::Result(const gl::Context *, gl::TextureType, egl::Image *));
    MOCK_METHOD1(generateMipmap, angle::Result(const gl::Context *));
    MOCK_METHOD2(bindTexImage, angle::Result(const gl::Context *, egl::Surface *));
    MOCK_METHOD1(releaseTexImage, angle::Result(const gl::Context *));

    MOCK_METHOD5(getAttachmentRenderTarget,
                 angle::Result(const gl::Context *,
                               GLenum,
                               const gl::ImageIndex &,
                               GLsizei,
                               FramebufferAttachmentRenderTarget **));

    MOCK_METHOD6(setStorageMultisample,
                 angle::Result(const gl::Context *,
                               gl::TextureType,
                               GLsizei,
                               GLint,
                               const gl::Extents &,
                               bool));

    MOCK_METHOD2(setBaseLevel, angle::Result(const gl::Context *, GLuint));

    MOCK_METHOD3(syncState,
                 angle::Result(const gl::Context *,
                               const gl::Texture::DirtyBits &,
                               gl::Command source));

    MOCK_METHOD0(destructor, void());

  protected:
    gl::TextureState mMockState;
};
}  // namespace rx

#endif  // LIBANGLE_RENDERER_TEXTUREIMPLMOCK_H_
