/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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

// SwapChain9.h: Defines a back-end specific class for the D3D9 swap chain.

#ifndef LIBANGLE_RENDERER_D3D_D3D9_SWAPCHAIN9_H_
#define LIBANGLE_RENDERER_D3D_D3D9_SWAPCHAIN9_H_

#include "common/angleutils.h"
#include "libANGLE/renderer/d3d/SwapChainD3D.h"
#include "libANGLE/renderer/d3d/d3d9/RenderTarget9.h"

namespace rx
{
class NativeWindow9;
class Renderer9;

class SwapChain9 : public SwapChainD3D
{
  public:
    SwapChain9(Renderer9 *renderer,
               NativeWindow9 *nativeWindow,
               HANDLE shareHandle,
               IUnknown *d3dTexture,
               GLenum backBufferFormat,
               GLenum depthBufferFormat,
               EGLint orientation);
    ~SwapChain9() override;

    EGLint resize(DisplayD3D *displayD3D, EGLint backbufferWidth, EGLint backbufferHeight) override;
    EGLint reset(DisplayD3D *displayD3D,
                 EGLint backbufferWidth,
                 EGLint backbufferHeight,
                 EGLint swapInterval) override;
    EGLint swapRect(DisplayD3D *displayD3D,
                    EGLint x,
                    EGLint y,
                    EGLint width,
                    EGLint height) override;
    void recreate() override;

    RenderTargetD3D *getColorRenderTarget() override;
    RenderTargetD3D *getDepthStencilRenderTarget() override;

    virtual IDirect3DSurface9 *getRenderTarget();
    virtual IDirect3DSurface9 *getDepthStencil();
    virtual IDirect3DTexture9 *getOffscreenTexture();

    EGLint getWidth() const { return mWidth; }
    EGLint getHeight() const { return mHeight; }

    void *getKeyedMutex() override;

    egl::Error getSyncValues(EGLuint64KHR *ust, EGLuint64KHR *msc, EGLuint64KHR *sbc) override;

  private:
    void release();

    Renderer9 *mRenderer;
    EGLint mWidth;
    EGLint mHeight;
    EGLint mSwapInterval;

    NativeWindow9 *mNativeWindow;

    IDirect3DSwapChain9 *mSwapChain;
    IDirect3DSurface9 *mBackBuffer;
    IDirect3DSurface9 *mRenderTarget;
    IDirect3DSurface9 *mDepthStencil;
    IDirect3DTexture9 *mOffscreenTexture;

    SurfaceRenderTarget9 mColorRenderTarget;
    SurfaceRenderTarget9 mDepthStencilRenderTarget;
};

}  // namespace rx
#endif  // LIBANGLE_RENDERER_D3D_D3D9_SWAPCHAIN9_H_
