/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#pragma once

namespace WebGPU {

class Buffer;
class Device;

// https://gpuweb.github.io/gpuweb/#gpucommandsmixin
class CommandsMixin {
protected PUBLIC_IN_WEBGPU_SWIFT:
    bool prepareTheEncoderState() const;
protected:
    NSString* encoderStateName() const;
    static bool computedSizeOverflows(const Buffer&, uint64_t offset, uint64_t& size);

    enum class EncoderState : uint8_t {
        Open,
        Locked,
        Ended
    };
    EncoderState m_state { EncoderState::Open };
};

} // namespace WebGPU
