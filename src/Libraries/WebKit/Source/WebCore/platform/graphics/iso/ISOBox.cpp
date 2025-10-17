/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 14, 2025.
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
#include "ISOBox.h"

#include <JavaScriptCore/DataView.h>
#include <wtf/TZoneMallocInlines.h>

using JSC::DataView;

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ISOBox);

ISOBox::ISOBox() = default;
ISOBox::~ISOBox() = default;
ISOBox::ISOBox(const ISOBox&) = default;

ISOBox::PeekResult ISOBox::peekBox(DataView& view, unsigned offset)
{
    unsigned maximumPossibleSize = view.byteLength() - offset;
    uint64_t size = 0;
    if (!checkedRead<uint32_t>(size, view, offset, BigEndian))
        return std::nullopt;

    FourCC type;
    if (!checkedRead<uint32_t>(type, view, offset, BigEndian))
        return std::nullopt;

    if (size == 1 && !checkedRead<uint64_t>(size, view, offset, BigEndian))
        return std::nullopt;

    if (size > maximumPossibleSize)
        size = maximumPossibleSize;
    else if (!size)
        size = maximumPossibleSize;

    return std::make_pair(type, size);
}

bool ISOBox::read(DataView& view)
{
    unsigned localOffset { 0 };
    return parse(view, localOffset);
}

bool ISOBox::read(DataView& view, unsigned& offset)
{
    unsigned localOffset = offset;
    if (!parse(view, localOffset))
        return false;

    offset += m_size;
    return true;
}

bool ISOBox::parse(DataView& view, unsigned& offset)
{
    unsigned maximumPossibleSize = view.byteLength() - offset;
    if (!checkedRead<uint32_t>(m_size, view, offset, BigEndian))
        return false;

    if (!checkedRead<uint32_t>(m_boxType, view, offset, BigEndian))
        return false;

    if (m_size == 1 && !checkedRead<uint64_t>(m_size, view, offset, BigEndian))
        return false;

    if (m_size > maximumPossibleSize)
        m_size = maximumPossibleSize;
    else if (!m_size)
        m_size = maximumPossibleSize;

    if (m_boxType == std::span { "uuid" }) {
        struct ExtendedType {
            uint8_t value[16];
        } extendedTypeStruct;
        if (!checkedRead<ExtendedType>(extendedTypeStruct, view, offset, BigEndian))
            return false;

        m_extendedType = Vector<uint8_t>(std::span { extendedTypeStruct.value });
    }

    return true;
}

ISOFullBox::ISOFullBox() = default;
ISOFullBox::ISOFullBox(const ISOFullBox&) = default;

bool ISOFullBox::parse(DataView& view, unsigned& offset)
{
    if (!ISOBox::parse(view, offset))
        return false;

    return parseVersionAndFlags(view, offset);
}

bool ISOFullBox::parseVersionAndFlags(DataView& view, unsigned& offset)
{
    uint32_t versionAndFlags = 0;
    if (!checkedRead<uint32_t>(versionAndFlags, view, offset, BigEndian))
        return false;

    m_version = versionAndFlags >> 24;
    m_flags = versionAndFlags & 0xFFFFFF;
    return true;
}

}
