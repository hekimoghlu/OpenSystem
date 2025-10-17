/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 23, 2021.
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

#include "LayoutUnit.h"
#include <wtf/OptionSet.h>

namespace WebCore {
namespace Layout {

struct HorizontalConstraints {
    LayoutUnit logicalRight() const { return logicalLeft + logicalWidth; }

    LayoutUnit logicalLeft;
    LayoutUnit logicalWidth;
};

struct VerticalConstraints {
    LayoutUnit logicalTop;
    LayoutUnit logicalHeight;
};

struct ConstraintsForInFlowContent {
    ConstraintsForInFlowContent(HorizontalConstraints, LayoutUnit logicalTop);

    HorizontalConstraints horizontal() const { return m_horizontal; }
    LayoutUnit logicalTop() const { return m_logicalTop; }

    enum BaseTypeFlag : uint8_t {
        GenericContent = 1 << 0,
        InlineContent  = 1 << 1,
        TableContent   = 1 << 2,
        FlexContent    = 1 << 3
    };
    bool isConstraintsForInlineContent() const { return baseTypeFlags().contains(InlineContent); }
    bool isConstraintsForTableContent() const { return baseTypeFlags().contains(TableContent); }
    bool isConstraintsForFlexContent() const { return baseTypeFlags().contains(FlexContent); }

protected:
    ConstraintsForInFlowContent(HorizontalConstraints, LayoutUnit logicalTop, OptionSet<BaseTypeFlag>);

private:
    OptionSet<BaseTypeFlag> baseTypeFlags() const { return OptionSet<BaseTypeFlag>::fromRaw(m_baseTypeFlags); }

    unsigned m_baseTypeFlags : 3; // OptionSet<BaseTypeFlag>
    HorizontalConstraints m_horizontal;
    LayoutUnit m_logicalTop;
};

struct ConstraintsForOutOfFlowContent {
    HorizontalConstraints horizontal;
    VerticalConstraints vertical;
    // Borders and padding are resolved against the containing block's content box as if the box was an in-flow box.
    LayoutUnit borderAndPaddingConstraints;
};

inline ConstraintsForInFlowContent::ConstraintsForInFlowContent(HorizontalConstraints horizontal, LayoutUnit logicalTop, OptionSet<BaseTypeFlag> baseTypeFlags)
    : m_baseTypeFlags(baseTypeFlags.toRaw())
    , m_horizontal(horizontal)
    , m_logicalTop(logicalTop)
{
}

inline ConstraintsForInFlowContent::ConstraintsForInFlowContent(HorizontalConstraints horizontal, LayoutUnit logicalTop)
    : ConstraintsForInFlowContent(horizontal, logicalTop, GenericContent)
{
}

enum class IntrinsicWidthMode {
    Minimum,
    Maximum
};

struct IntrinsicWidthConstraints {
    void expand(LayoutUnit horizontalValue);
    IntrinsicWidthConstraints& operator+=(const IntrinsicWidthConstraints&);
    IntrinsicWidthConstraints& operator+=(LayoutUnit);
    IntrinsicWidthConstraints& operator-=(const IntrinsicWidthConstraints&);
    IntrinsicWidthConstraints& operator-=(LayoutUnit);

    LayoutUnit minimum;
    LayoutUnit maximum;
};

inline void IntrinsicWidthConstraints::expand(LayoutUnit horizontalValue)
{
    minimum += horizontalValue;
    maximum += horizontalValue;
}

inline IntrinsicWidthConstraints& IntrinsicWidthConstraints::operator+=(const IntrinsicWidthConstraints& other)
{
    minimum += other.minimum;
    maximum += other.maximum;
    return *this;
}

inline IntrinsicWidthConstraints& IntrinsicWidthConstraints::operator+=(LayoutUnit value)
{
    expand(value);
    return *this;
}

inline IntrinsicWidthConstraints& IntrinsicWidthConstraints::operator-=(const IntrinsicWidthConstraints& other)
{
    minimum -= other.minimum;
    maximum -= other.maximum;
    return *this;
}

inline IntrinsicWidthConstraints& IntrinsicWidthConstraints::operator-=(LayoutUnit value)
{
    expand(-value);
    return *this;
}

}
}

#define SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONSTRAINTS(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Layout::ToValueTypeName) \
    static bool isType(const WebCore::Layout::ConstraintsForInFlowContent& constraints) { return constraints.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

