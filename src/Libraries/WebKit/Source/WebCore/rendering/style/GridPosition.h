/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

enum class GridPositionType : uint8_t {
    AutoPosition,
    ExplicitPosition, // [ <integer> || <string> ]
    SpanPosition, // span && [ <integer> || <string> ]
    NamedGridAreaPosition // <ident>
};

enum class GridPositionSide : uint8_t {
    ColumnStartSide,
    ColumnEndSide,
    RowStartSide,
    RowEndSide
};

class GridPosition {
public:
    bool isPositive() const { return integerPosition() > 0; }

    GridPositionType type() const { return m_type; }
    bool isAuto() const { return m_type == GridPositionType::AutoPosition; }
    bool isSpan() const { return m_type == GridPositionType::SpanPosition; }
    bool isNamedGridArea() const { return m_type == GridPositionType::NamedGridAreaPosition; }

    WEBCORE_EXPORT void setExplicitPosition(int position, const String& namedGridLine);
    void setAutoPosition();
    WEBCORE_EXPORT void setSpanPosition(int position, const String& namedGridLine);
    void setNamedGridArea(const String&);

    WEBCORE_EXPORT int integerPosition() const;
    String namedGridLine() const;
    WEBCORE_EXPORT int spanPosition() const;

    friend bool operator==(const GridPosition&, const GridPosition&) = default;

    bool shouldBeResolvedAgainstOppositePosition() const { return isAuto() || isSpan(); }

    // Note that grid line 1 is internally represented by the index 0, that's why the max value for
    // a position is kGridMaxTracks instead of kGridMaxTracks + 1.
    static int max();
    static int min();

    WEBCORE_EXPORT static void setMaxPositionForTesting(unsigned);

private:
    static std::optional<int> gMaxPositionForTesting;

    void setIntegerPosition(int integerPosition) { m_integerPosition = clampTo(integerPosition, min(), max()); }

    GridPositionType m_type { GridPositionType::AutoPosition };
    int m_integerPosition { 0 };
    String m_namedGridLine;
};

WTF::TextStream& operator<<(WTF::TextStream&, const GridPosition&);

} // namespace WebCore
