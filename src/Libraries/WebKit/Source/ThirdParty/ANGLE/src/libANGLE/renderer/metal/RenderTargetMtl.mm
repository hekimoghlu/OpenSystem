/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 12, 2021.
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
// RenderTargetMtl.mm:
//    Implements the class methods for RenderTargetMtl.
//

#include "libANGLE/renderer/metal/RenderTargetMtl.h"

namespace rx
{
RenderTargetMtl::RenderTargetMtl() {}

RenderTargetMtl::~RenderTargetMtl()
{
    reset();
}

void RenderTargetMtl::set(const mtl::TextureRef &texture,
                          const mtl::MipmapNativeLevel &level,
                          uint32_t layer,
                          const mtl::Format &format)
{
    setWithImplicitMSTexture(texture, nullptr, level, layer, format);
}

void RenderTargetMtl::setWithImplicitMSTexture(const mtl::TextureRef &texture,
                                               const mtl::TextureRef &implicitMSTexture,
                                               const mtl::MipmapNativeLevel &level,
                                               uint32_t layer,
                                               const mtl::Format &format)
{
    mTexture           = texture;
    mImplicitMSTexture = implicitMSTexture;
    mLevelIndex        = level;
    mLayerIndex        = layer;
    mFormat            = format;
}

void RenderTargetMtl::setTexture(const mtl::TextureRef &texture)
{
    mTexture = texture;
}

void RenderTargetMtl::setImplicitMSTexture(const mtl::TextureRef &implicitMSTexture)
{
    mImplicitMSTexture = implicitMSTexture;
}

void RenderTargetMtl::duplicateFrom(const RenderTargetMtl &src)
{
    setWithImplicitMSTexture(src.getTexture(), src.getImplicitMSTexture(), src.getLevelIndex(),
                             src.getLayerIndex(), src.getFormat());
}

void RenderTargetMtl::reset()
{
    mTexture.reset();
    mImplicitMSTexture.reset();
    mLevelIndex = mtl::kZeroNativeMipLevel;
    mLayerIndex = 0;
    mFormat     = mtl::Format();
}

uint32_t RenderTargetMtl::getRenderSamples() const
{
    mtl::TextureRef implicitMSTex = getImplicitMSTexture();
    mtl::TextureRef tex           = getTexture();
    return implicitMSTex ? implicitMSTex->samples() : (tex ? tex->samples() : 1);
}

void RenderTargetMtl::toRenderPassAttachmentDesc(mtl::RenderPassAttachmentDesc *rpaDescOut) const
{
    mtl::TextureRef implicitMSTex = getImplicitMSTexture();
    mtl::TextureRef tex           = getTexture();
    if (implicitMSTex)
    {
        rpaDescOut->texture             = implicitMSTex;
        rpaDescOut->resolveTexture      = tex;
        rpaDescOut->resolveLevel        = mLevelIndex;
        rpaDescOut->resolveSliceOrDepth = mLayerIndex;
    }
    else
    {
        rpaDescOut->texture      = tex;
        rpaDescOut->level        = mLevelIndex;
        rpaDescOut->sliceOrDepth = mLayerIndex;
    }
    rpaDescOut->blendable = mFormat.getCaps().blendable;
}

#if ANGLE_WEBKIT_EXPLICIT_RESOLVE_TARGET_ENABLED
void RenderTargetMtl::toRenderPassResolveAttachmentDesc(
    mtl::RenderPassAttachmentDesc *rpaDescOut) const
{
    ASSERT(!getImplicitMSTexture());
    ASSERT(getRenderSamples() == 1);
    rpaDescOut->resolveTexture      = getTexture();
    rpaDescOut->resolveLevel        = mLevelIndex;
    rpaDescOut->resolveSliceOrDepth = mLayerIndex;
}
#endif
}  // namespace rx
