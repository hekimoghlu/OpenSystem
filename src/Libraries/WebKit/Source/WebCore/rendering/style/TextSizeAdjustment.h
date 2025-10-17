/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 5, 2025.
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

#if ENABLE(TEXT_AUTOSIZING)

#include <wtf/OptionSet.h>

namespace WebCore {

class RenderStyle;

class TextSizeAdjustment {
public:
    static constexpr TextSizeAdjustment autoAdjust() { return TextSizeAdjustment(true); }
    static constexpr TextSizeAdjustment none() { return TextSizeAdjustment(false); }

    constexpr TextSizeAdjustment() : m_value(Auto) { }
    constexpr TextSizeAdjustment(float value) : m_value(value) { ASSERT_UNDER_CONSTEXPR_CONTEXT(m_value >= 0); }

    constexpr float percentage() const { ASSERT_UNDER_CONSTEXPR_CONTEXT(m_value >= 0); return m_value; }
    constexpr float multiplier() const { ASSERT_UNDER_CONSTEXPR_CONTEXT(m_value >= 0); return m_value / 100; }

    constexpr bool isAuto() const { return m_value == Auto; }
    constexpr bool isNone() const { return m_value == None; }
    constexpr bool isPercentage() const { return m_value >= 0; }

    friend constexpr bool operator==(TextSizeAdjustment, TextSizeAdjustment) = default;

private:
    static constexpr float Auto = -1;
    static constexpr float None = -2;
    constexpr TextSizeAdjustment(bool isAuto) : m_value(isAuto ? Auto : None) { }
    float m_value;
};

class AutosizeStatus {
public:
    enum class Fields : uint8_t {
        AvoidSubtree = 1 << 0,
        FixedHeight = 1 << 1,
        FixedWidth = 1 << 2,
        Floating = 1 << 3,
        OverflowXHidden = 1 << 4,
        // Adding new values requires giving RenderStyle::InheritedFlags::autosizeStatus additional bits.
    };

    constexpr AutosizeStatus(OptionSet<Fields>);
    constexpr OptionSet<Fields> fields() const { return m_fields; }

    constexpr bool contains(Fields) const;

    friend constexpr bool operator==(AutosizeStatus, AutosizeStatus) = default;

    static float idempotentTextSize(float specifiedSize, float pageScale);
    static AutosizeStatus computeStatus(const RenderStyle&);
    static void updateStatus(RenderStyle&);

    static bool probablyContainsASmallFixedNumberOfLines(const RenderStyle&);

private:
    OptionSet<Fields> m_fields;
};

constexpr AutosizeStatus::AutosizeStatus(OptionSet<Fields> fields)
    : m_fields(fields)
{
}

constexpr bool AutosizeStatus::contains(Fields fields) const
{
    return m_fields.contains(fields);
}

} // namespace WebCore

#endif // ENABLE(TEXT_AUTOSIZING)
