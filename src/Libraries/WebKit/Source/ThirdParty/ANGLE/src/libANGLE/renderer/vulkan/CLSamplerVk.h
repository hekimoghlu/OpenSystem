/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 29, 2022.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// CLSamplerVk.h: Defines the class interface for CLSamplerVk, implementing CLSamplerImpl.

#ifndef LIBANGLE_RENDERER_VULKAN_CLSAMPLERVK_H_
#define LIBANGLE_RENDERER_VULKAN_CLSAMPLERVK_H_

#include "clspv/Sampler.h"
#include "libANGLE/renderer/CLSamplerImpl.h"
#include "libANGLE/renderer/vulkan/cl_types.h"
#include "libANGLE/renderer/vulkan/vk_cache_utils.h"
#include "vulkan/vulkan_core.h"

namespace rx
{

class CLSamplerVk : public CLSamplerImpl
{
  public:
    CLSamplerVk(const cl::Sampler &sampler);
    ~CLSamplerVk() override;

    vk::SamplerHelper &getSamplerHelper() { return mSamplerHelper; }
    vk::SamplerHelper &getSamplerHelperNormalized() { return mSamplerHelperNormalized; }
    angle::Result create();
    angle::Result createNormalized();

    VkSamplerAddressMode getVkAddressMode();
    VkFilter getVkFilter();
    VkSamplerMipmapMode getVkMipmapMode();
    uint32_t getSamplerMask();

  private:
    CLContextVk *mContext;
    vk::Renderer *mRenderer;

    vk::SamplerHelper mSamplerHelper;
    vk::SamplerHelper mSamplerHelperNormalized;
    VkSamplerCreateInfo mDefaultSamplerCreateInfo;
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_VULKAN_CLSAMPLERVK_H_
