/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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

#include "FontMetrics.h"
#include "Length.h"
#include <wtf/text/TextStream.h>

namespace WebCore {

class TextDecorationThickness {
public:
    TextDecorationThickness() = default;

    static TextDecorationThickness createWithAuto()
    {
        return TextDecorationThickness(Type::Auto);
    }
    static TextDecorationThickness createFromFont()
    {
        return TextDecorationThickness(Type::FromFont);
    }
    static TextDecorationThickness createWithLength(Length&& length)
    {
        return { Type::Length, WTFMove(length) };
    }

    constexpr bool isAuto() const
    {
        return m_type == Type::Auto;
    }

    constexpr bool isFromFont() const
    {
        return m_type == Type::FromFont;
    }

    constexpr bool isLength() const
    {
        return m_type == Type::Length;
    }

    const Length& length() const
    {
        ASSERT(isLength());
        return m_length;
    }

    float resolve(float fontSize, const FontMetrics& metrics) const
    {
        if (isAuto()) {
            const float textDecorationBaseFontSize = 16;
            return fontSize / textDecorationBaseFontSize;
        }
        if (isFromFont())
            return metrics.underlineThickness().value_or(0);

        ASSERT(isLength());
        if (m_length.isPercent())
            return fontSize * (m_length.percent() / 100.0f);
        if (m_length.isCalculated())
            return m_length.nonNanCalculatedValue(fontSize);
        return m_length.value();
    }

    constexpr bool operator==(const TextDecorationThickness& other) const
    {
        switch (m_type) {
        case Type::Auto:
        case Type::FromFont:
            return m_type == other.m_type;
        case Type::Length:
            return m_type == other.m_type && m_length == other.m_length;
        default:
            ASSERT_NOT_REACHED();
            return true;
        }
    }

private:
    enum class Type : uint8_t {
        Auto,
        FromFont,
        Length
    };

    TextDecorationThickness(Type type)
        : m_type(type)
    {
    }

    TextDecorationThickness(Type type, Length&& length)
        : m_type(type)
        , m_length(WTFMove(length))
    {
    }

    Type m_type { };
    Length m_length { };
};

inline TextStream& operator<<(TextStream& ts, const TextDecorationThickness& thickness)
{
    if (thickness.isAuto())
        ts << "auto";
    else if (thickness.isFromFont())
        ts << "from-font";
    else
        ts << thickness.length();
    return ts;
}

} // namespace WebCore
