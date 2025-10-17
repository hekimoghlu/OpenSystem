/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 22, 2022.
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
#ifndef OpenTypeTypes_h
#define OpenTypeTypes_h

#if ENABLE(OPENTYPE_MATH)
#include "Glyph.h"
#endif

#include "SharedBuffer.h"
#include <wtf/StdLibExtras.h>

namespace WebCore {
namespace OpenType {

struct BigEndianShort {
    operator short() const { return (v & 0x00ff) << 8 | v >> 8; }
    BigEndianShort(short u) : v((u & 0x00ff) << 8 | u >> 8) { }
    unsigned short v;
};

struct BigEndianUShort {
    operator unsigned short() const { return (v & 0x00ff) << 8 | v >> 8; }
    BigEndianUShort(unsigned short u) : v((u & 0x00ff) << 8 | u >> 8) { }
    unsigned short v;
};

struct BigEndianLong {
    operator int() const { return (v & 0xff) << 24 | (v & 0xff00) << 8 | (v & 0xff0000) >> 8 | v >> 24; }
    BigEndianLong(int u) : v((u & 0xff) << 24 | (u & 0xff00) << 8 | (u & 0xff0000) >> 8 | u >> 24) { }
    unsigned v;
};

struct BigEndianULong {
    operator unsigned() const { return (v & 0xff) << 24 | (v & 0xff00) << 8 | (v & 0xff0000) >> 8 | v >> 24; }
    BigEndianULong(unsigned u) : v((u & 0xff) << 24 | (u & 0xff00) << 8 | (u & 0xff0000) >> 8 | u >> 24) { }
    unsigned v;
};

typedef BigEndianShort Int16;
typedef BigEndianUShort UInt16;
typedef BigEndianLong Int32;
typedef BigEndianULong UInt32;

typedef UInt32 Fixed;
typedef UInt16 Offset;
typedef UInt16 GlyphID;

// OTTag is native because it's only compared against constants, so we don't
// do endian conversion here but make sure constants are in big-endian order.
// Note that multi-character literal is implementation-defined in C++0x.
typedef uint32_t Tag;
#define OT_MAKE_TAG(ch1, ch2, ch3, ch4) ((((uint32_t)(ch4)) << 24) | (((uint32_t)(ch3)) << 16) | (((uint32_t)(ch2)) << 8) | ((uint32_t)(ch1)))

template<typename T> static const T* validateTableSingle(const RefPtr<SharedBuffer>& buffer)
{
    if (!buffer || buffer->size() < sizeof(T))
        return nullptr;
    return &reinterpretCastSpanStartTo<const T>(buffer->span());
}

template<typename T> static std::span<const T> validateTable(const RefPtr<SharedBuffer>& buffer, size_t count)
{
    if (!buffer || (buffer->size() / sizeof(T)) < count)
        return { };
    return spanReinterpretCast<const T>(buffer->span().first(sizeof(T) * count));
}

struct TableBase {
protected:
    static bool isValidEnd(const SharedBuffer& buffer, const void* position)
    {
        auto bufferSpan = buffer.span();
        if (position < bufferSpan.data())
            return false;
        size_t offset = static_cast<const uint8_t*>(position) - bufferSpan.data();
        return offset <= buffer.size(); // "<=" because end is included as valid
    }

WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    template <typename T> static const T* validatePtr(const SharedBuffer& buffer, const void* position)
    {
        const T* casted = reinterpret_cast<const T*>(position);
        if (!isValidEnd(buffer, &casted[1]))
            return 0;
        return casted;
    }

    template <typename T> const T* validateOffset(const SharedBuffer& buffer, uint16_t offset) const
    {
        return validatePtr<T>(buffer, reinterpret_cast<const int8_t*>(this) + offset);
    }
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
};

#if ENABLE(OPENTYPE_VERTICAL) || ENABLE(OPENTYPE_MATH)
struct CoverageTable : TableBase {
    OpenType::UInt16 coverageFormat;
};

struct Coverage1Table : CoverageTable {
    OpenType::UInt16 glyphCount;
    OpenType::GlyphID glyphArray[1];
};

struct Coverage2Table : CoverageTable {
    OpenType::UInt16 rangeCount;
    struct RangeRecord {
        OpenType::GlyphID start;
        OpenType::GlyphID end;
        OpenType::UInt16 startCoverageIndex;
    } ranges[1];
};
#endif // ENABLE(OPENTYPE_VERTICAL) || ENABLE(OPENTYPE_MATH)

#if ENABLE(OPENTYPE_MATH)
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
struct TableWithCoverage : TableBase {
protected:
    bool getCoverageIndex(const SharedBuffer& buffer, const CoverageTable* coverage, Glyph glyph, uint32_t& coverageIndex) const
    {
        switch (coverage->coverageFormat) {
        case 1: { // Coverage Format 1
            const Coverage1Table* coverage1 = validatePtr<Coverage1Table>(buffer, coverage);
            if (!coverage1)
                return false;
            uint16_t glyphCount = coverage1->glyphCount;
            if (!isValidEnd(buffer, &coverage1->glyphArray[glyphCount]))
                return false;

            // We do a binary search on the glyph indexes.
            uint32_t imin = 0, imax = glyphCount;
            while (imin < imax) {
                uint32_t imid = (imin + imax) >> 1;
                uint16_t glyphMid = coverage1->glyphArray[imid];
                if (glyphMid == glyph) {
                    coverageIndex = imid;
                    return true;
                }
                if (glyphMid < glyph)
                    imin = imid + 1;
                else
                    imax = imid;
            }
            break;
        }
        case 2: { // Coverage Format 2
            const Coverage2Table* coverage2 = validatePtr<Coverage2Table>(buffer, coverage);
            if (!coverage2)
                return false;
            uint16_t rangeCount = coverage2->rangeCount;
            if (!isValidEnd(buffer, &coverage2->ranges[rangeCount]))
                return false;

            // We do a binary search on the ranges.
            uint32_t imin = 0, imax = rangeCount;
            while (imin < imax) {
                uint32_t imid = (imin + imax) >> 1;
                uint16_t rStart = coverage2->ranges[imid].start;
                uint16_t rEnd = coverage2->ranges[imid].end;
                if (rEnd < glyph)
                    imin = imid + 1;
                else if (glyph < rStart)
                    imax = imid;
                else {
                    coverageIndex = coverage2->ranges[imid].startCoverageIndex + glyph - rStart;
                    return true;
                }
            }
            break;
        }
        }
        return false;
    }
};
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
#endif

} // namespace OpenType
} // namespace WebCore

#endif // OpenTypeTypes_h
