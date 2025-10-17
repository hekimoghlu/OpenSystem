/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 15, 2024.
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
#include <TargetConditionals.h>

#if !TARGET_OS_EXCLAVEKIT

#include <bit>

#include "Defines.h"
#include "BitUtils.h"
#include "PVLEInt64.h"

namespace lsl {
void emitPVLEUInt64(uint64_t value, Vector<std::byte>& data) {
    auto valueBytes = (std::byte*)&value;
    const uint8_t activeBits = std::max<uint8_t>(lsl::bit_width(value),1);
    if (activeBits > 56) {
        data.push_back((std::byte)0);
        std::copy(&valueBytes[0], &valueBytes[8], std::back_inserter(data));
        return;
    }
    const uint8_t bytes = (activeBits+6)/7;
    value <<= bytes;
    value |= 1<<(bytes-1);
    std::copy(&valueBytes[0], &valueBytes[bytes], std::back_inserter(data));
}

bool readPVLEUInt64(std::span<std::byte>& data, uint64_t& result) {
    result = 0;
    if (data.size() == 0) {
        return false;
    }
    const uint8_t additionalByteCount = std::countr_zero((uint8_t)data[0]);
    if (data.size() < additionalByteCount+1) {
        return false;
    }
    if (additionalByteCount == 8) {
        std::copy(data.begin()+1, data.begin()+9, (std::byte*)&result);
        data = data.last(data.size()-9);
        return true;
    }
    const uint8_t extraBitCount     = 8 - (additionalByteCount+1);
    const uint8_t extraBits         = (((uint8_t)(data[0]))>>(additionalByteCount+1)) & ((1<<extraBitCount)-1);
    std::copy(data.begin()+1, data.begin()+additionalByteCount+1, (std::byte*)&result);
    result <<= extraBitCount;
    result |= extraBits;
    data = data.last(data.size()-(additionalByteCount+1));
    return true;
}

};

#endif // !TARGET_OS_EXCLAVEKIT
