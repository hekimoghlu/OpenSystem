/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
// RenderTargetMtl.h:
//    Defines the class interface for RenderTargetMtl.
//

#ifndef LIBANGLE_RENDERER_METAL_RENDERTARGETMTL_H_
#define LIBANGLE_RENDERER_METAL_RENDERTARGETMTL_H_

#import <Metal/Metal.h>

#include "libANGLE/FramebufferAttachment.h"
#include "libANGLE/renderer/metal/mtl_format_utils.h"
#include "libANGLE/renderer/metal/mtl_resources.h"
#include "libANGLE/renderer/metal/mtl_state_cache.h"

namespace rx
{

// This is a very light-weight class that does not own to the resources it points to.
// It's meant only to copy across some information from a FramebufferAttachment to the
// business rendering logic.
class RenderTargetMtl final : public FramebufferAttachmentRenderTarget
{
  public:
    RenderTargetMtl();
    ~RenderTargetMtl() override;

    void set(const mtl::TextureRef &texture,
             const mtl::MipmapNativeLevel &level,
             uint32_t layer,
             const mtl::Format &format);
    void setWithImplicitMSTexture(const mtl::TextureRef &texture,
                                  const mtl::TextureRef &implicitMSTexture,
                                  const mtl::MipmapNativeLevel &level,
                                  uint32_t layer,
                                  const mtl::Format &format);
    void setTexture(const mtl::TextureRef &texture);
    void setImplicitMSTexture(const mtl::TextureRef &implicitMSTexture);
    void duplicateFrom(const RenderTargetMtl &src);
    void reset();

    mtl::TextureRef getTexture() const { return mTexture.lock(); }
    mtl::TextureRef getImplicitMSTexture() const { return mImplicitMSTexture.lock(); }
    const mtl::MipmapNativeLevel &getLevelIndex() const { return mLevelIndex; }
    uint32_t getLayerIndex() const { return mLayerIndex; }
    uint32_t getRenderSamples() const;
    const mtl::Format &getFormat() const { return mFormat; }

    void toRenderPassAttachmentDesc(mtl::RenderPassAttachmentDesc *rpaDescOut) const;
#if ANGLE_WEBKIT_EXPLICIT_RESOLVE_TARGET_ENABLED
    void toRenderPassResolveAttachmentDesc(mtl::RenderPassAttachmentDesc *rpaDescOut) const;
#endif

  private:
    mtl::TextureWeakRef mTexture;
    mtl::TextureWeakRef mImplicitMSTexture;
    mtl::MipmapNativeLevel mLevelIndex = mtl::kZeroNativeMipLevel;
    uint32_t mLayerIndex               = 0;
    mtl::Format mFormat;
};
}  // namespace rx

#endif /* LIBANGLE_RENDERER_METAL_RENDERTARGETMTL_H */
