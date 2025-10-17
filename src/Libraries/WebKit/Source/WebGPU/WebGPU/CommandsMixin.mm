/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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
#import "config.h"
#import "CommandsMixin.h"

#import "Buffer.h"
#import "Device.h"
#import "PipelineLayout.h"
#import "RenderPipeline.h"

namespace WebGPU {

bool CommandsMixin::prepareTheEncoderState() const
{
    // https://gpuweb.github.io/gpuweb/#abstract-opdef-prepare-the-encoder-state

    switch (m_state) {
    case EncoderState::Open:
        return true;
    case EncoderState::Locked:
        // FIXME: "Make encoder invalid"
        return false;
    case EncoderState::Ended:
        // FIXME: "Generate a validation error"
        return false;
    }
}

NSString* CommandsMixin::encoderStateName() const
{
    switch (m_state) {
    case EncoderState::Open:
        return @"Open";
    case EncoderState::Locked:
        return @"Locked";
    case EncoderState::Ended:
        return @"Ended";
    }
}

bool CommandsMixin::computedSizeOverflows(const Buffer& buffer, uint64_t offset, uint64_t& size)
{
    if (size == WGPU_WHOLE_SIZE) {
        auto localSize = checkedDifference<uint64_t>(buffer.initialSize(), offset);
        if (localSize.hasOverflowed())
            return true;

        size = localSize.value();
    }

    auto sum = checkedSum<uint64_t>(offset, size);
    if (sum.hasOverflowed() || sum.value() > buffer.initialSize())
        return true;

    return false;
}

} // namespace WebGPU
