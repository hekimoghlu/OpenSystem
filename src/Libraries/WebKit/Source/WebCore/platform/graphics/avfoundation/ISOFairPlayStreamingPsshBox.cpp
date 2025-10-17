/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#include "ISOFairPlayStreamingPsshBox.h"

#include <JavaScriptCore/DataView.h>
#include <wtf/StdLibExtras.h>

namespace WebCore {

const Vector<uint8_t>& ISOFairPlayStreamingPsshBox::fairPlaySystemID()
{
    static NeverDestroyed<Vector<uint8_t>> systemID = Vector<uint8_t>({ 0x94, 0xCE, 0x86, 0xFB, 0x07, 0xFF, 0x4F, 0x43, 0xAD, 0xB8, 0x93, 0xD2, 0xFA, 0x96, 0x8C, 0xA2 });
    return systemID;
}

bool ISOFairPlayStreamingInfoBox::parse(JSC::DataView& view, unsigned& offset)
{
    if (!ISOFullBox::parse(view, offset))
        return false;

    return checkedRead<uint32_t>(m_scheme, view, offset, BigEndian);
}

bool ISOFairPlayStreamingKeyRequestInfoBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    CheckedUint64 remaining = m_size;
    remaining -= (localOffset - offset);
    if (remaining.hasOverflowed())
        return false;

    if (remaining < m_keyID.capacity())
        return false;

    auto buffer = view.possiblySharedBuffer();
    if (!buffer)
        return false;

    auto keyID = buffer->slice(localOffset, localOffset + m_keyID.capacity());
    localOffset += m_keyID.capacity();

    m_keyID.resize(m_keyID.capacity());
    if (keyID->byteLength() < m_keyID.capacity())
        return false;
    memcpySpan(m_keyID.mutableSpan(), keyID->span().first(m_keyID.capacity()));

    offset = localOffset;
    return true;
}

bool ISOFairPlayStreamingKeyAssetIdBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    if (localOffset - offset == m_size) {
        m_data.clear();
        offset = localOffset;
        return true;
    }

    auto buffer = view.possiblySharedBuffer();
    if (!buffer)
        return false;

    size_t dataSize;
    if (!WTF::safeSub(m_size, localOffset - offset, dataSize))
        return false;

    auto parsedData = buffer->slice(localOffset, localOffset + dataSize);
    localOffset += dataSize;

    m_data.resize(dataSize);
    if (parsedData->byteLength() < dataSize)
        return false;
    memcpySpan(m_data.mutableSpan(), parsedData->span().first(dataSize));
    offset = localOffset;
    return true;
}

bool ISOFairPlayStreamingKeyContextBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    if (localOffset - offset == m_size) {
        m_data.clear();
        offset = localOffset;
        return true;
    }

    auto buffer = view.possiblySharedBuffer();
    if (!buffer)
        return false;

    size_t dataSize;
    if (!WTF::safeSub(m_size, localOffset - offset, dataSize))
        return false;

    auto parsedData = buffer->slice(localOffset, localOffset + dataSize);
    localOffset += dataSize;

    m_data.resize(dataSize);
    if (parsedData->byteLength() < dataSize)
        return false;
    memcpySpan(m_data.mutableSpan(), parsedData->span().first(dataSize));
    offset = localOffset;
    return true;
}

bool ISOFairPlayStreamingKeyVersionListBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    do {
        if (localOffset - offset == m_size)
            break;

        uint64_t remaining;
        if (!WTF::safeSub(m_size, localOffset - offset, remaining))
            return false;

        if (remaining < sizeof(uint32_t))
            return false;

        uint32_t version;
        if (!checkedRead<uint32_t>(version, view, localOffset, BigEndian))
            return false;
        m_versions.append(version);
    } while (true);

    offset = localOffset;
    return true; 
}

bool ISOFairPlayStreamingKeyRequestBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    if (!m_requestInfo.read(view, localOffset))
        return false;

    while (localOffset - offset < m_size) {
        auto result = peekBox(view, localOffset);
        if (!result)
            return false;

        auto name = result.value().first;
        if (name == ISOFairPlayStreamingKeyAssetIdBox::boxTypeName()) {
            if (m_assetID)
                return false;

            ISOFairPlayStreamingKeyAssetIdBox assetID;
            if (!assetID.read(view, localOffset))
                return false;

            m_assetID = WTFMove(assetID);
            continue;
        }

        if (name == ISOFairPlayStreamingKeyContextBox::boxTypeName()) {
            if (m_context)
                return false;

            ISOFairPlayStreamingKeyContextBox context;
            if (!context.read(view, localOffset))
                return false;

            m_context = WTFMove(context);
            continue;
        }

        if (name == ISOFairPlayStreamingKeyVersionListBox::boxTypeName()) {
            if (m_versionList)
                return false;

            ISOFairPlayStreamingKeyVersionListBox versionList;
            if (!versionList.read(view, localOffset))
                return false;

            m_versionList = WTFMove(versionList);
            continue;
        }

        // Unknown box type; error.
        return false;
    }   
    
    offset = localOffset;
    return true; 
}

bool ISOFairPlayStreamingInitDataBox::parse(JSC::DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!ISOBox::parse(view, localOffset))
        return false;

    if (!m_info.read(view, localOffset))
        return false;

    while (localOffset - offset < m_size) {
        ISOFairPlayStreamingKeyRequestBox request;
        if (!request.read(view, localOffset))
            return false;

        m_requests.append(WTFMove(request));
    }

    offset = localOffset;
    return true;
}

bool ISOFairPlayStreamingPsshBox::parse(JSC::DataView& view, unsigned& offset)
{
    if (!ISOProtectionSystemSpecificHeaderBox::parse(view, offset))
        return false;

    // Back up the offset by exactly the size of m_data:
    offset -= m_data.size();

    return m_initDataBox.read(view, offset);
}

ISOFairPlayStreamingInfoBox::ISOFairPlayStreamingInfoBox() = default;
ISOFairPlayStreamingInfoBox::ISOFairPlayStreamingInfoBox(const ISOFairPlayStreamingInfoBox&) = default;
ISOFairPlayStreamingInfoBox::~ISOFairPlayStreamingInfoBox() = default;

ISOFairPlayStreamingKeyRequestInfoBox::ISOFairPlayStreamingKeyRequestInfoBox() = default;
ISOFairPlayStreamingKeyRequestInfoBox::~ISOFairPlayStreamingKeyRequestInfoBox() = default;

ISOFairPlayStreamingKeyAssetIdBox::ISOFairPlayStreamingKeyAssetIdBox() = default;
ISOFairPlayStreamingKeyAssetIdBox::ISOFairPlayStreamingKeyAssetIdBox(const ISOFairPlayStreamingKeyAssetIdBox&) = default;
ISOFairPlayStreamingKeyAssetIdBox::~ISOFairPlayStreamingKeyAssetIdBox() = default;

ISOFairPlayStreamingKeyContextBox::ISOFairPlayStreamingKeyContextBox() = default;
ISOFairPlayStreamingKeyContextBox::ISOFairPlayStreamingKeyContextBox(const ISOFairPlayStreamingKeyContextBox&) = default;
ISOFairPlayStreamingKeyContextBox::~ISOFairPlayStreamingKeyContextBox() = default;

ISOFairPlayStreamingKeyVersionListBox::ISOFairPlayStreamingKeyVersionListBox() = default;
ISOFairPlayStreamingKeyVersionListBox::ISOFairPlayStreamingKeyVersionListBox(const ISOFairPlayStreamingKeyVersionListBox&) = default;
ISOFairPlayStreamingKeyVersionListBox::~ISOFairPlayStreamingKeyVersionListBox() = default;

ISOFairPlayStreamingKeyRequestBox::ISOFairPlayStreamingKeyRequestBox() = default;
ISOFairPlayStreamingKeyRequestBox::ISOFairPlayStreamingKeyRequestBox(const ISOFairPlayStreamingKeyRequestBox&) = default;
ISOFairPlayStreamingKeyRequestBox::~ISOFairPlayStreamingKeyRequestBox() = default;

ISOFairPlayStreamingInitDataBox::ISOFairPlayStreamingInitDataBox() = default;
ISOFairPlayStreamingInitDataBox::~ISOFairPlayStreamingInitDataBox() = default;

ISOFairPlayStreamingPsshBox::ISOFairPlayStreamingPsshBox() = default;
ISOFairPlayStreamingPsshBox::~ISOFairPlayStreamingPsshBox() = default;

} // namespace WebCore
