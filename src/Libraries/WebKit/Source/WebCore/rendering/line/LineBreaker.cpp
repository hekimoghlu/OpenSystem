/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 13, 2025.
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
#include "LineBreaker.h"

#include "BreakingContext.h"

namespace WebCore {

void LineBreaker::skipTrailingWhitespace(LegacyInlineIterator& iterator, const LineInfo& lineInfo)
{
    while (!iterator.atEnd() && !requiresLineBox(iterator, lineInfo, TrailingWhitespace))
        iterator.increment();
}

void LineBreaker::skipLeadingWhitespace(InlineBidiResolver& resolver, LineInfo& lineInfo)
{
    while (!resolver.position().atEnd() && !requiresLineBox(resolver.position(), lineInfo, LeadingWhitespace))
        resolver.increment();

    resolver.commitExplicitEmbedding();
}

LegacyInlineIterator LineBreaker::nextLineBreak(InlineBidiResolver& resolver, LineInfo& lineInfo, RenderTextInfo& renderTextInfo)
{
    ASSERT(resolver.position().root() == &m_block);

    bool appliedStartWidth = resolver.position().offset();

    LineWidth width(m_block, lineInfo.isFirstLine());

    skipLeadingWhitespace(resolver, lineInfo);

    if (resolver.position().atEnd())
        return resolver.position();

    BreakingContext context(*this, resolver, lineInfo, width, renderTextInfo, appliedStartWidth, m_block);

    while (context.currentObject()) {
        context.initializeForCurrentObject();
        if (context.currentObject()->isRenderInline()) {
            context.handleEmptyInline();
        } else if (context.currentObject()->isRenderText()) {
            if (context.handleText()) {
                // We've hit a hard text line break. Our line break iterator is updated, so early return.
                return context.lineBreak();
            }
        } else if (context.currentObject()->isLineBreakOpportunity())
            context.commitLineBreakAtCurrentWidth(*context.currentObject());
        else
            ASSERT_NOT_REACHED();

        if (context.atEnd())
            return context.handleEndOfLine();

        context.commitAndUpdateLineBreakIfNeeded();

        if (context.atEnd())
            return context.handleEndOfLine();

        context.increment();
    }

    context.clearLineBreakIfFitsOnLine(true);

    return context.handleEndOfLine();
}

}
