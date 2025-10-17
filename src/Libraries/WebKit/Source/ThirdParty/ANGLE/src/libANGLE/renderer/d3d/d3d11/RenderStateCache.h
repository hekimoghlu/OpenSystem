/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

// RenderStateCache.h: Defines rx::RenderStateCache, a cache of Direct3D render
// state objects.

#ifndef LIBANGLE_RENDERER_D3D_D3D11_RENDERSTATECACHE_H_
#define LIBANGLE_RENDERER_D3D_D3D11_RENDERSTATECACHE_H_

#include "common/angleutils.h"
#include "libANGLE/Error.h"
#include "libANGLE/SizedMRUCache.h"
#include "libANGLE/angletypes.h"
#include "libANGLE/renderer/d3d/d3d11/renderer11_utils.h"

#include <unordered_map>

namespace std
{
template <>
struct hash<rx::d3d11::BlendStateKey>
{
    size_t operator()(const rx::d3d11::BlendStateKey &key) const
    {
        return angle::ComputeGenericHash(key);
    }
};

template <>
struct hash<rx::d3d11::RasterizerStateKey>
{
    size_t operator()(const rx::d3d11::RasterizerStateKey &key) const
    {
        return angle::ComputeGenericHash(key);
    }
};

template <>
struct hash<gl::DepthStencilState>
{
    size_t operator()(const gl::DepthStencilState &key) const
    {
        return angle::ComputeGenericHash(key);
    }
};

template <>
struct hash<gl::SamplerState>
{
    size_t operator()(const gl::SamplerState &key) const { return angle::ComputeGenericHash(key); }
};
}  // namespace std

namespace rx
{
class Framebuffer11;
class Renderer11;

class RenderStateCache : angle::NonCopyable
{
  public:
    RenderStateCache();
    virtual ~RenderStateCache();

    void clear();

    static d3d11::BlendStateKey GetBlendStateKey(const gl::Context *context,
                                                 Framebuffer11 *framebuffer11,
                                                 const gl::BlendStateExt &blendStateExt,
                                                 bool sampleAlphaToCoverage);
    angle::Result getBlendState(const gl::Context *context,
                                Renderer11 *renderer,
                                const d3d11::BlendStateKey &key,
                                const d3d11::BlendState **outBlendState);
    angle::Result getRasterizerState(const gl::Context *context,
                                     Renderer11 *renderer,
                                     const gl::RasterizerState &rasterState,
                                     bool scissorEnabled,
                                     ID3D11RasterizerState **outRasterizerState);
    angle::Result getDepthStencilState(const gl::Context *context,
                                       Renderer11 *renderer,
                                       const gl::DepthStencilState &dsState,
                                       const d3d11::DepthStencilState **outDSState);
    angle::Result getSamplerState(const gl::Context *context,
                                  Renderer11 *renderer,
                                  const gl::SamplerState &samplerState,
                                  ID3D11SamplerState **outSamplerState);

  private:
    // MSDN's documentation of ID3D11Device::CreateBlendState, ID3D11Device::CreateRasterizerState,
    // ID3D11Device::CreateDepthStencilState and ID3D11Device::CreateSamplerState claims the maximum
    // number of unique states of each type an application can create is 4096
    // TODO(ShahmeerEsmail): Revisit the cache sizes to make sure they are appropriate for most
    // scenarios.
    static constexpr unsigned int kMaxStates = 4096;

    // The cache tries to clean up this many states at once.
    static constexpr unsigned int kGCLimit = 128;

    // Blend state cache
    using BlendStateMap = angle::base::HashingMRUCache<d3d11::BlendStateKey, d3d11::BlendState>;
    BlendStateMap mBlendStateCache;

    // Rasterizer state cache
    using RasterizerStateMap =
        angle::base::HashingMRUCache<d3d11::RasterizerStateKey, d3d11::RasterizerState>;
    RasterizerStateMap mRasterizerStateCache;

    // Depth stencil state cache
    using DepthStencilStateMap =
        angle::base::HashingMRUCache<gl::DepthStencilState, d3d11::DepthStencilState>;
    DepthStencilStateMap mDepthStencilStateCache;

    // Sample state cache
    using SamplerStateMap = angle::base::HashingMRUCache<gl::SamplerState, d3d11::SamplerState>;
    SamplerStateMap mSamplerStateCache;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_D3D_D3D11_RENDERSTATECACHE_H_
