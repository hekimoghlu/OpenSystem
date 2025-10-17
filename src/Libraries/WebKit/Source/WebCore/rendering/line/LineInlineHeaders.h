/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 24, 2022.
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

#include "LineInfo.h"
#include "RenderBoxInlines.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderLayer.h"
#include "RenderObjectInlines.h"

namespace WebCore {

enum WhitespacePosition : bool { LeadingWhitespace, TrailingWhitespace };

inline const RenderStyle& lineStyle(const RenderObject& renderer, const LineInfo& lineInfo)
{
    return lineInfo.isFirstLine() ? renderer.firstLineStyle() : renderer.style();
}

inline bool requiresLineBoxForContent(const RenderInline& flow, const LineInfo& lineInfo)
{
    RenderElement* parent = flow.parent();
    if (flow.document().inNoQuirksMode()) {
        const RenderStyle& flowStyle = lineStyle(flow, lineInfo);
        const RenderStyle& parentStyle = lineStyle(*parent, lineInfo);
        if (flowStyle.lineHeight() != parentStyle.lineHeight()
            || flowStyle.verticalAlign() != parentStyle.verticalAlign()
            || !parentStyle.fontCascade().metricsOfPrimaryFont().hasIdenticalAscentDescentAndLineGap(flowStyle.fontCascade().metricsOfPrimaryFont()))
        return true;
    }
    return false;
}

inline bool shouldCollapseWhiteSpace(const RenderStyle* style, const LineInfo& lineInfo, WhitespacePosition whitespacePosition)
{
    // CSS2 16.6.1
    // If a space (U+0020) at the beginning of a line has 'white-space' set to 'normal', 'nowrap', or 'pre-line', it is removed.
    // If a space (U+0020) at the end of a line has 'white-space' set to 'normal', 'nowrap', or 'pre-line', it is also removed.
    // If spaces (U+0020) or tabs (U+0009) at the end of a line have 'white-space' set to 'pre-wrap', UAs may visually collapse them.
    return style->collapseWhiteSpace()
        || (whitespacePosition == TrailingWhitespace && style->whiteSpaceCollapse() == WhiteSpaceCollapse::Preserve
            && style->textWrapMode() == TextWrapMode::Wrap && !lineInfo.isEmpty());
}

inline bool skipNonBreakingSpace(const LegacyInlineIterator& it, const LineInfo& lineInfo)
{
    if (it.renderer()->style().nbspMode() != NBSPMode::Space || it.current() != noBreakSpace)
        return false;

    // FIXME: This is bad. It makes nbsp inconsistent with space and won't work correctly
    // with m_minWidth/m_maxWidth.
    // Do not skip a non-breaking space if it is the first character
    // on a line after a clean line break (or on the first line, since previousLineBrokeCleanly starts off
    // |true|).
    if (lineInfo.isEmpty())
        return false;

    return true;
}

inline bool requiresLineBox(const LegacyInlineIterator& it, const LineInfo& lineInfo = LineInfo(), WhitespacePosition whitespacePosition = LeadingWhitespace)
{
    bool rendererIsEmptyInline = false;
    if (auto* inlineRenderer = dynamicDowncast<RenderInline>(*it.renderer())) {
        if (!requiresLineBoxForContent(*inlineRenderer, lineInfo))
            return false;
        rendererIsEmptyInline = isEmptyInline(*inlineRenderer);
    }

    if (!shouldCollapseWhiteSpace(&it.renderer()->style(), lineInfo, whitespacePosition))
        return true;

    UChar current = it.current();
    bool notJustWhitespace = current != ' ' && current != '\t' && current != softHyphen && (current != '\n' || it.renderer()->preservesNewline()) && !skipNonBreakingSpace(it, lineInfo);
    return notJustWhitespace || rendererIsEmptyInline;
}

} // namespace WebCore
