/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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
// Copyright 2012 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// RenderTarget9.h: Defines a D3D9-specific wrapper for IDirect3DSurface9 pointers
// retained by Renderbuffers.

#ifndef LIBANGLE_RENDERER_D3D_D3D9_RENDERTARGET9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_RENDERTARGET9_H_

#include "libANGLE/renderer/d3d/RenderTargetD3D.h"

namespace rx
{
class Renderer9;
class SwapChain9;

class RenderTarget9 : public RenderTargetD3D
{
  public:
    RenderTarget9() {}
    ~RenderTarget9() override {}
    // Retrieve the texture that backs this render target, may be null for swap chain render
    // targets.
    virtual IDirect3DBaseTexture9 *getTexture() const = 0;
    virtual size_t getTextureLevel() const            = 0;

    virtual IDirect3DSurface9 *getSurface() const = 0;

    virtual D3DFORMAT getD3DFormat() const = 0;
};

class TextureRenderTarget9 : public RenderTarget9
{
  public:
    TextureRenderTarget9(IDirect3DBaseTexture9 *texture,
                         size_t textureLevel,
                         IDirect3DSurface9 *surface,
                         GLenum internalFormat,
                         GLsizei width,
                         GLsizei height,
                         GLsizei depth,
                         GLsizei samples);
    ~TextureRenderTarget9() override;

    GLsizei getWidth() const override;
    GLsizei getHeight() const override;
    GLsizei getDepth() const override;
    GLenum getInternalFormat() const override;
    GLsizei getSamples() const override;

    IDirect3DBaseTexture9 *getTexture() const override;
    size_t getTextureLevel() const override;
    IDirect3DSurface9 *getSurface() const override;

    D3DFORMAT getD3DFormat() const override;

  private:
    GLsizei mWidth;
    GLsizei mHeight;
    GLsizei mDepth;
    GLenum mInternalFormat;
    D3DFORMAT mD3DFormat;
    GLsizei mSamples;

    IDirect3DBaseTexture9 *mTexture;
    size_t mTextureLevel;
    IDirect3DSurface9 *mRenderTarget;
};

class SurfaceRenderTarget9 : public RenderTarget9
{
  public:
    SurfaceRenderTarget9(SwapChain9 *swapChain, bool depth);
    ~SurfaceRenderTarget9() override;

    GLsizei getWidth() const override;
    GLsizei getHeight() const override;
    GLsizei getDepth() const override;
    GLenum getInternalFormat() const override;
    GLsizei getSamples() const override;

    IDirect3DBaseTexture9 *getTexture() const override;
    size_t getTextureLevel() const override;
    IDirect3DSurface9 *getSurface() const override;

    D3DFORMAT getD3DFormat() const override;

  private:
    SwapChain9 *mSwapChain;
    bool mDepth;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D9_RENDERTARGET9_H_
