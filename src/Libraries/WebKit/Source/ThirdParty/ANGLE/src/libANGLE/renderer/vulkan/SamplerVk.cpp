/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
// SamplerVk.cpp:
//    Implements the class methods for SamplerVk.
//

#include "libANGLE/renderer/vulkan/SamplerVk.h"

#include "common/debug.h"
#include "libANGLE/Context.h"
#include "libANGLE/renderer/vulkan/vk_utils.h"

namespace rx
{

SamplerVk::SamplerVk(const gl::SamplerState &state) : SamplerImpl(state) {}

SamplerVk::~SamplerVk() = default;

void SamplerVk::onDestroy(const gl::Context *context)
{
    mSampler.reset();
}

angle::Result SamplerVk::syncState(const gl::Context *context, const bool dirty)
{
    ContextVk *contextVk = vk::GetImpl(context);

    vk::Renderer *renderer = contextVk->getRenderer();
    if (mSampler)
    {
        if (!dirty)
        {
            return angle::Result::Continue;
        }
        mSampler.reset();
    }

    vk::SamplerDesc desc(contextVk, mState, false, nullptr, static_cast<angle::FormatID>(0));
    ANGLE_TRY(renderer->getSamplerCache().getSampler(contextVk, desc, &mSampler));

    return angle::Result::Continue;
}

}  // namespace rx
