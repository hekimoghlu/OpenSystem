/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 13, 2023.
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
#include "ISOTrackEncryptionBox.h"

#include <JavaScriptCore/DataView.h>
#include <wtf/StdLibExtras.h>

using JSC::DataView;

namespace WebCore {

ISOTrackEncryptionBox::ISOTrackEncryptionBox() = default;
ISOTrackEncryptionBox::~ISOTrackEncryptionBox() = default;

bool ISOTrackEncryptionBox::parseWithoutTypeAndSize(DataView& view)
{
    // Clients may want to parse the contents of a `tenc` box without the
    // leading size and name fields.
    unsigned offset = 0;
    return parseVersionAndFlags(view, offset) && parsePayload(view, offset);
}

bool ISOTrackEncryptionBox::parse(DataView& view, unsigned& offset)
{
    // ISO/IEC 23001-7-2015 Section 8.2.2
    if (!ISOFullBox::parse(view, offset))
        return false;

    return parsePayload(view, offset);
}

bool ISOTrackEncryptionBox::parsePayload(DataView& view, unsigned& offset)
{
    // unsigned int(8) reserved = 0;
    offset += 1;

    if (!m_version) {
        // unsigned int(8) reserved = 0;
        offset += 1;
    } else {
        int8_t cryptAndSkip = 0;
        if (!checkedRead<int8_t>(cryptAndSkip, view, offset, BigEndian))
            return false;

        m_defaultCryptByteBlock = cryptAndSkip >> 4;
        m_defaultSkipByteBlock = cryptAndSkip & 0xF;
    }

    if (!checkedRead<int8_t>(m_defaultIsProtected, view, offset, BigEndian))
        return false;

    if (!checkedRead<int8_t>(m_defaultPerSampleIVSize, view, offset, BigEndian))
        return false;

    auto buffer = view.possiblySharedBuffer();
    if (!buffer)
        return false;

    auto keyIDBuffer = buffer->slice(offset, offset + 16);
    offset += 16;

    m_defaultKID.resize(16);
    if (keyIDBuffer->byteLength() < 16)
        return false;

    memcpySpan(m_defaultKID.mutableSpan(), keyIDBuffer->span().first(16));

    if (m_defaultIsProtected == 1 && !m_defaultPerSampleIVSize) {
        int8_t defaultConstantIVSize = 0;
        if (!checkedRead<int8_t>(defaultConstantIVSize, view, offset, BigEndian) || defaultConstantIVSize < 0)
            return false;

        Vector<uint8_t> defaultConstantIV;
        defaultConstantIV.reserveInitialCapacity(defaultConstantIVSize);
        while (defaultConstantIVSize--) {
            int8_t character = 0;
            if (!checkedRead<int8_t>(character, view, offset, BigEndian))
                return false;
            defaultConstantIV.append(character);
        }
        m_defaultConstantIV = WTFMove(defaultConstantIV);
    }

    return true;
}

}
