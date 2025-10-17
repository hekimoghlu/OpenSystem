/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//

// TransformFeedbackD3D.cpp is a no-op implementation for both the D3D9 and D3D11 renderers.

#include "libANGLE/renderer/d3d/d3d11/TransformFeedback11.h"

#include "libANGLE/Buffer.h"
#include "libANGLE/renderer/d3d/d3d11/Buffer11.h"
#include "libANGLE/renderer/d3d/d3d11/Renderer11.h"

namespace rx
{

TransformFeedback11::TransformFeedback11(const gl::TransformFeedbackState &state,
                                         Renderer11 *renderer)
    : TransformFeedbackImpl(state),
      mRenderer(renderer),
      mIsDirty(true),
      mBuffers(state.getIndexedBuffers().size(), nullptr),
      mBufferOffsets(state.getIndexedBuffers().size(), 0),
      mSerial(mRenderer->generateSerial())
{}

TransformFeedback11::~TransformFeedback11() {}

angle::Result TransformFeedback11::begin(const gl::Context *context,
                                         gl::PrimitiveMode primitiveMode)
{
    // Reset all the cached offsets to the binding offsets
    mIsDirty = true;
    for (size_t bindingIdx = 0; bindingIdx < mBuffers.size(); bindingIdx++)
    {
        const auto &binding = mState.getIndexedBuffer(bindingIdx);
        if (binding.get() != nullptr)
        {
            mBufferOffsets[bindingIdx] = static_cast<UINT>(binding.getOffset());
        }
        else
        {
            mBufferOffsets[bindingIdx] = 0;
        }
    }
    mRenderer->getStateManager()->invalidateTransformFeedback();
    return angle::Result::Continue;
}

angle::Result TransformFeedback11::end(const gl::Context *context)
{
    mRenderer->getStateManager()->invalidateTransformFeedback();
    if (mRenderer->getFeatures().flushAfterEndingTransformFeedback.enabled)
    {
        mRenderer->getDeviceContext()->Flush();
    }
    return angle::Result::Continue;
}

angle::Result TransformFeedback11::pause(const gl::Context *context)
{
    mRenderer->getStateManager()->invalidateTransformFeedback();
    return angle::Result::Continue;
}

angle::Result TransformFeedback11::resume(const gl::Context *context)
{
    mRenderer->getStateManager()->invalidateTransformFeedback();
    return angle::Result::Continue;
}

angle::Result TransformFeedback11::bindIndexedBuffer(
    const gl::Context *context,
    size_t index,
    const gl::OffsetBindingPointer<gl::Buffer> &binding)
{
    mIsDirty              = true;
    mBufferOffsets[index] = static_cast<UINT>(binding.getOffset());
    mRenderer->getStateManager()->invalidateTransformFeedback();
    return angle::Result::Continue;
}

void TransformFeedback11::onApply()
{
    mIsDirty = false;

    // Change all buffer offsets to -1 so that if any of them need to be re-applied, the are set to
    // append
    std::fill(mBufferOffsets.begin(), mBufferOffsets.end(), -1);
}

bool TransformFeedback11::isDirty() const
{
    return mIsDirty;
}

UINT TransformFeedback11::getNumSOBuffers() const
{
    return static_cast<UINT>(mBuffers.size());
}

angle::Result TransformFeedback11::getSOBuffers(const gl::Context *context,
                                                const std::vector<ID3D11Buffer *> **buffersOut)
{
    for (size_t bindingIdx = 0; bindingIdx < mBuffers.size(); bindingIdx++)
    {
        const auto &binding = mState.getIndexedBuffer(bindingIdx);
        if (binding.get() != nullptr)
        {
            Buffer11 *storage = GetImplAs<Buffer11>(binding.get());
            ANGLE_TRY(storage->getBuffer(context, BUFFER_USAGE_VERTEX_OR_TRANSFORM_FEEDBACK,
                                         &mBuffers[bindingIdx]));
        }
    }

    *buffersOut = &mBuffers;
    return angle::Result::Continue;
}

const std::vector<UINT> &TransformFeedback11::getSOBufferOffsets() const
{
    return mBufferOffsets;
}

UniqueSerial TransformFeedback11::getSerial() const
{
    return mSerial;
}

}  // namespace rx
