/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 7, 2024.
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

#include "FourCC.h"
#include <wtf/Forward.h>
#include <wtf/StdIntExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/TypeCasts.h>

namespace JSC {
class DataView;
}

namespace WebCore {

class ISOBox {
    WTF_MAKE_TZONE_ALLOCATED(ISOBox);
public:
    WEBCORE_EXPORT ISOBox();
    WEBCORE_EXPORT ISOBox(const ISOBox&);
    ISOBox(ISOBox&&) = default;
    WEBCORE_EXPORT virtual ~ISOBox();

    ISOBox& operator=(const ISOBox&) = default;
    ISOBox& operator=(ISOBox&&) = default;

    using PeekResult = std::optional<std::pair<FourCC, uint64_t>>;
    static PeekResult peekBox(JSC::DataView&, unsigned offset);
    static constexpr size_t minimumBoxSize() { return 2 * sizeof(uint32_t); }

    WEBCORE_EXPORT bool read(JSC::DataView&);
    bool read(JSC::DataView&, unsigned& offset);

    uint64_t size() const { return m_size; }
    FourCC boxType() const { return m_boxType; }
    const Vector<uint8_t>& extendedType() const { return m_extendedType; }

protected:
    virtual bool parse(JSC::DataView&, unsigned& offset);

    enum Endianness { BigEndian, LittleEndian };

    template<typename T, typename R, typename V>
    static bool checkedRead(R& returnValue, V& view, unsigned& offset, Endianness endianness)
    {
        bool readStatus = false;
        size_t actualOffset = offset;
        T value = view.template read<T>(actualOffset, endianness == LittleEndian, &readStatus);
        RELEASE_ASSERT(isInBounds<uint32_t>(actualOffset));
        offset = actualOffset;
        if (!readStatus)
            return false;

        returnValue = value;
        return true;
    }

    uint64_t m_size { 0 };
    FourCC m_boxType;
    Vector<uint8_t> m_extendedType;
};

class ISOFullBox : public ISOBox {
public:
    WEBCORE_EXPORT ISOFullBox();
    WEBCORE_EXPORT ISOFullBox(const ISOFullBox&);
    ISOFullBox(ISOFullBox&&) = default;

    uint8_t version() const { return m_version; }
    uint32_t flags() const { return m_flags; }

protected:
    bool parse(JSC::DataView&, unsigned& offset) override;
    bool parseVersionAndFlags(JSC::DataView&, unsigned& offset);

    uint8_t m_version { 0 };
    uint32_t m_flags { 0 };
};

}

#define SPECIALIZE_TYPE_TRAITS_ISOBOX(ISOBoxType) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::ISOBoxType) \
static bool isType(const WebCore::ISOBox& box) { return box.boxType() == WebCore::ISOBoxType::boxTypeName(); } \
SPECIALIZE_TYPE_TRAITS_END()
