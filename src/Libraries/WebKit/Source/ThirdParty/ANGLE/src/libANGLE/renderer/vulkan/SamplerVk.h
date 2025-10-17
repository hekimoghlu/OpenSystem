/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
// Copyright 2016 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// SamplerVk.h:
//    Defines the class interface for SamplerVk, implementing SamplerImpl.
//

#ifndef LIBANGLE_RENDERER_VULKAN_SAMPLERVK_H_
#define LIBANGLE_RENDERER_VULKAN_SAMPLERVK_H_

#include "libANGLE/renderer/SamplerImpl.h"
#include "libANGLE/renderer/vulkan/ContextVk.h"
#include "libANGLE/renderer/vulkan/vk_helpers.h"

namespace rx
{

class SamplerVk : public SamplerImpl
{
  public:
    SamplerVk(const gl::SamplerState &state);
    ~SamplerVk() override;

    void onDestroy(const gl::Context *context) override;
    angle::Result syncState(const gl::Context *context, const bool dirty) override;

    const vk::SamplerHelper &getSampler() const
    {
        ASSERT(mSampler);
        ASSERT(mSampler->valid());
        return *mSampler.get();
    }

  private:
    vk::SharedSamplerPtr mSampler;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_SAMPLERVK_H_
