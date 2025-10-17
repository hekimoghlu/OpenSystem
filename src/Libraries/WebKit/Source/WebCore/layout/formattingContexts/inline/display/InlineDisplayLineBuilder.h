/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 29, 2024.
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

#include "InlineDisplayLine.h"
#include "InlineFormattingContext.h"
#include "InlineLineBuilder.h"

namespace WebCore {
namespace Layout {

class InlineLayoutState;
class LineBox;

class InlineDisplayLineBuilder {
public:
    InlineDisplayLineBuilder(InlineFormattingContext&, const ConstraintsForInlineContent&);

    InlineDisplay::Line build(const LineLayoutResult&, const LineBox&, bool lineIsFullyTruncatedInBlockDirection) const;

    static void applyEllipsisIfNeeded(LineEndingTruncationPolicy, InlineDisplay::Line&, InlineDisplay::Boxes&, bool isLegacyLineClamp);
    static void addLegacyLineClampTrailingLinkBoxIfApplicable(const InlineFormattingContext&, const InlineLayoutState&, InlineDisplay::Content&);

private:
    struct EnclosingLineGeometry {
        InlineDisplay::Line::EnclosingTopAndBottom enclosingTopAndBottom;
        InlineRect contentOverflowRect;
    };
    EnclosingLineGeometry collectEnclosingLineGeometry(const LineLayoutResult&, const LineBox&, const InlineRect& lineBoxRect) const;

    const ConstraintsForInlineContent& constraints() const { return m_constraints; }
    const InlineFormattingContext& formattingContext() const { return m_inlineFormattingContext; }
    InlineFormattingContext& formattingContext() { return m_inlineFormattingContext; }
    const Box& root() const { return formattingContext().root(); }

private:
    InlineFormattingContext& m_inlineFormattingContext;
    const ConstraintsForInlineContent& m_constraints;
};

}
}

