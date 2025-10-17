/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
// Copyright 2013 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// Clear11.h: Framebuffer clear utility class.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_CLEAR11_H_
#define LIBANGLE_RENDERER_D3D_D3D11_CLEAR11_H_

#include <map>
#include <vector>

#include "libANGLE/Error.h"
#include "libANGLE/Framebuffer.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/d3d/d3d11/renderer11_utils.h"

namespace rx
{
class Renderer11;
class RenderTarget11;
struct ClearParameters;

template <typename T>
struct RtvDsvClearInfo
{
    T r, g, b, a;
    float z;
    float c1padding[3];
};

class Clear11 : angle::NonCopyable
{
  public:
    explicit Clear11(Renderer11 *renderer);
    ~Clear11();

    // Clears the framebuffer with the supplied clear parameters, assumes that the framebuffer is
    // currently applied.
    angle::Result clearFramebuffer(const gl::Context *context,
                                   const ClearParameters &clearParams,
                                   const gl::FramebufferState &fboData);

  private:
    class ShaderManager final : angle::NonCopyable
    {
      public:
        ShaderManager();
        ~ShaderManager();
        angle::Result getShadersAndLayout(const gl::Context *context,
                                          Renderer11 *renderer,
                                          const INT clearType,
                                          const uint32_t numRTs,
                                          const bool hasLayeredLayout,
                                          const d3d11::InputLayout **il,
                                          const d3d11::VertexShader **vs,
                                          const d3d11::GeometryShader **gs,
                                          const d3d11::PixelShader **ps);

      private:
        constexpr static size_t kNumShaders = D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT;

        d3d11::LazyShader<ID3D11VertexShader> mVs;
        d3d11::LazyShader<ID3D11VertexShader> mVsMultiview;
        d3d11::LazyShader<ID3D11GeometryShader> mGsMultiview;
        d3d11::LazyShader<ID3D11PixelShader> mPsDepth;
        std::array<d3d11::LazyShader<ID3D11PixelShader>, kNumShaders> mPsFloat;
        std::array<d3d11::LazyShader<ID3D11PixelShader>, kNumShaders> mPsUInt;
        std::array<d3d11::LazyShader<ID3D11PixelShader>, kNumShaders> mPsSInt;
    };

    bool useVertexBuffer() const;
    angle::Result ensureConstantBufferCreated(const gl::Context *context);
    angle::Result ensureVertexBufferCreated(const gl::Context *context);
    angle::Result ensureResourcesInitialized(const gl::Context *context);

    Renderer11 *mRenderer;
    bool mResourcesInitialized;

    // States
    d3d11::RasterizerState mScissorEnabledRasterizerState;
    d3d11::RasterizerState mScissorDisabledRasterizerState;
    gl::DepthStencilState mDepthStencilStateKey;
    d3d11::BlendStateKey mBlendStateKey;

    // Shaders and shader resources
    ShaderManager mShaderManager;
    d3d11::Buffer mConstantBuffer;
    d3d11::Buffer mVertexBuffer;

    // Buffer data and draw parameters
    RtvDsvClearInfo<float> mShaderData;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_CLEAR11_H_
