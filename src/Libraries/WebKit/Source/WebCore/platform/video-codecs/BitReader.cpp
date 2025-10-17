/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 14, 2023.
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
#include "config.h"
#include "BitReader.h"

namespace WebCore {

std::optional<uint64_t> BitReader::read(size_t bits)
{
    ASSERT(bits <= 64);

    // FIXME: We should optimize this routine.
    size_t value = 0;
    do {
        auto bit = readBit();
        if (!bit)
            return { };
        value = (value << 1) | (*bit ? 1 : 0);
        --bits;
    } while (bits);
    return value;
}

std::optional<bool> BitReader::readBit()
{
    if (!m_remainingBits) {
        if (m_index >= m_data.size())
            return { };
        m_currentByte = m_data[m_index++];
        m_remainingBits = 8;
    }
    
    bool value = m_currentByte & 0x80;
    --m_remainingBits;
    m_currentByte = m_currentByte << 1;
    return value;
}

}
