/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#include "ISOProtectionSystemSpecificHeaderBox.h"

#include <JavaScriptCore/DataView.h>
#include <wtf/StdLibExtras.h>

using JSC::DataView;

namespace WebCore {

ISOProtectionSystemSpecificHeaderBox::ISOProtectionSystemSpecificHeaderBox() = default;
ISOProtectionSystemSpecificHeaderBox::~ISOProtectionSystemSpecificHeaderBox() = default;

std::optional<Vector<uint8_t>> ISOProtectionSystemSpecificHeaderBox::peekSystemID(JSC::DataView& view, unsigned offset)
{
    auto peekResult = ISOBox::peekBox(view, offset);
    if (!peekResult || peekResult.value().first != boxTypeName())
        return std::nullopt;

    ISOProtectionSystemSpecificHeaderBox psshBox;
    psshBox.parse(view, offset);
    return psshBox.systemID();
}

bool ISOProtectionSystemSpecificHeaderBox::parse(DataView& view, unsigned& offset)
{
    if (!ISOFullBox::parse(view, offset))
        return false;

    // ISO/IEC 23001-7-2016 Section 8.1.1
    auto buffer = view.possiblySharedBuffer();
    if (!buffer)
        return false;
    auto systemID = buffer->slice(offset, offset + 16);
    offset += 16;

    m_systemID.resize(16);
    if (systemID->byteLength() < 16)
        return false;

    memcpySpan(m_systemID.mutableSpan(), systemID->span().first(16));

    if (m_version) {
        uint32_t keyIDCount = 0;
        if (!checkedRead<uint32_t>(keyIDCount, view, offset, BigEndian))
            return false;
        if (buffer->byteLength() - offset < keyIDCount * 16)
            return false;
        if (!m_keyIDs.tryReserveCapacity(keyIDCount))
            return false;
        m_keyIDs.resize(keyIDCount);
        for (unsigned keyID = 0; keyID < keyIDCount; keyID++) {
            auto& currentKeyID = m_keyIDs[keyID];
            currentKeyID.resize(16);
            auto parsedKeyID = buffer->slice(offset, offset + 16);
            offset += 16;
            if (parsedKeyID->byteLength() < 16)
                continue;
            memcpySpan(currentKeyID.mutableSpan(), parsedKeyID->span().first(16));
        }
    }

    uint32_t dataSize = 0;
    if (!checkedRead<uint32_t>(dataSize, view, offset, BigEndian))
        return false;
    if (buffer->byteLength() - offset < dataSize)
        return false;
    auto parsedData = buffer->slice(offset, offset + dataSize);
    offset += dataSize;

    m_data.resize(dataSize);
    if (parsedData->byteLength() < dataSize)
        return false;

    memcpySpan(m_data.mutableSpan(), parsedData->span().first(dataSize));

    return true;
}

}
