/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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
#include <variant>
#include <wtf/Markable.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

struct FontSizeAdjust {
    friend bool operator==(const FontSizeAdjust&, const FontSizeAdjust&) = default;

    enum class ValueType : bool { Number, FromFont };
    enum class Metric : uint8_t {
        ExHeight,
        CapHeight,
        ChWidth,
        IcWidth,
        IcHeight
    };

    std::optional<float> resolve(float computedSize, const FontMetrics& fontMetrics) const
    {
        std::optional<float> metricValue;
        switch (metric) {
        case FontSizeAdjust::Metric::CapHeight:
            metricValue = fontMetrics.capHeight();
            break;
        case FontSizeAdjust::Metric::ChWidth:
            metricValue = fontMetrics.zeroWidth();
            break;
        // FIXME: Are ic-height and ic-width the same? Gecko treats them the same.
        case FontSizeAdjust::Metric::IcWidth:
        case FontSizeAdjust::Metric::IcHeight:
            metricValue = fontMetrics.ideogramWidth();
            break;
        case FontSizeAdjust::Metric::ExHeight:
        default:
            metricValue = fontMetrics.xHeight();
        }

        return metricValue.has_value() && computedSize
            ? std::make_optional(*metricValue / computedSize)
            : std::nullopt;
    }

    bool isNone() const { return !value && type != ValueType::FromFont; }
    bool isFromFont() const { return type == ValueType::FromFont; }
    bool shouldResolveFromFont() const { return isFromFont() && !value; }

    Metric metric { Metric::ExHeight };
    ValueType type { ValueType::Number };
    Markable<float, WTF::FloatMarkableTraits> value { };
};

inline void add(Hasher& hasher, const FontSizeAdjust& fontSizeAdjust)
{
    add(hasher, fontSizeAdjust.metric, fontSizeAdjust.type, fontSizeAdjust.value.unsafeValue());
}

inline TextStream& operator<<(TextStream& ts, const FontSizeAdjust& fontSizeAdjust)
{
    switch (fontSizeAdjust.metric) {
    case FontSizeAdjust::Metric::CapHeight:
        ts << "cap-height";
        break;
    case FontSizeAdjust::Metric::ChWidth:
        ts << "ch-width";
        break;
    case FontSizeAdjust::Metric::IcWidth:
        ts << "ic-width";
        break;
    case FontSizeAdjust::Metric::IcHeight:
        ts << "ic-height";
        break;
    case FontSizeAdjust::Metric::ExHeight:
    default:
        if (fontSizeAdjust.isFromFont())
            return ts << "from-font";
        return ts << *fontSizeAdjust.value;
    }

    if (fontSizeAdjust.isFromFont())
        return ts << " " << "from-font";
    return ts << " " << *fontSizeAdjust.value;
}

}
