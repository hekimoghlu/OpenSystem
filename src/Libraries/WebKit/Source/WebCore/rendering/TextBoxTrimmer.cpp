/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
#include "TextBoxTrimmer.h"

#include "InlineIteratorBox.h"
#include "InlineIteratorLineBox.h"
#include "RenderBoxModelObjectInlines.h"
#include "RenderMultiColumnFlow.h"
#include "RenderView.h"

namespace WebCore {

static TextBoxTrim textBoxTrim(const RenderBlockFlow& textBoxTrimRoot)
{
    if (auto* multiColumnFlow = dynamicDowncast<RenderMultiColumnFlow>(textBoxTrimRoot))
        return multiColumnFlow->multiColumnBlockFlow()->style().textBoxTrim();
    return textBoxTrimRoot.style().textBoxTrim();
}

static void removeTextBoxTrimStart(LocalFrameViewLayoutContext& layoutContext)
{
    auto textBoxTrim = layoutContext.textBoxTrim();
    if (!textBoxTrim || !textBoxTrim->trimFirstFormattedLine) {
        ASSERT_NOT_REACHED();
        return;
    }
    layoutContext.setTextBoxTrim(LocalFrameViewLayoutContext::TextBoxTrim { false, textBoxTrim->lastFormattedLineRoot });
}

static bool shouldIgnoreAsFirstLastFormattedLineContainer(const RenderBlockFlow& container)
{
    if (container.style().display() == DisplayType::RubyAnnotation || container.createsNewFormattingContext())
        return true;
    // Empty continuation pre/post blocks should be ignored as they are implementation detail.
    if (container.isAnonymousBlock()) {
        if (auto firstLineBox = InlineIterator::firstLineBoxFor(container))
            return !firstLineBox->lineLeftmostLeafBox();
        return true;
    }
    return false;
}

static inline RenderBlockFlow* firstFormattedLineRoot(const RenderBlockFlow& enclosingBlockContainer)
{
    for (auto* child = enclosingBlockContainer.firstChild(); child; child = child->nextSibling()) {
        CheckedPtr blockContainer = dynamicDowncast<RenderBlockFlow>(*child);
        if (!blockContainer || blockContainer->createsNewFormattingContext())
            continue;
        if (blockContainer->hasLines())
            return blockContainer.get();
        if (auto* descendantRoot = firstFormattedLineRoot(*blockContainer))
            return descendantRoot;
        if (!shouldIgnoreAsFirstLastFormattedLineContainer(*blockContainer))
            return nullptr;
    }
    return nullptr;
}

static RenderBlockFlow* lastFormattedLineRoot(const RenderBlockFlow& enclosingBlockContainer)
{
    for (auto* child = enclosingBlockContainer.lastChild(); child; child = child->previousSibling()) {
        CheckedPtr blockContainer = dynamicDowncast<RenderBlockFlow>(*child);
        if (!blockContainer || blockContainer->createsNewFormattingContext())
            continue;
        if (blockContainer->hasLines())
            return blockContainer.get();
        if (auto* descendantRoot = lastFormattedLineRoot(*blockContainer))
            return descendantRoot;
        if (!shouldIgnoreAsFirstLastFormattedLineContainer(*blockContainer))
            return nullptr;
    }
    return nullptr;
}

TextBoxTrimmer::TextBoxTrimmer(const RenderBlockFlow& blockContainer)
    : m_blockContainer(blockContainer)
{
    adjustTextBoxTrimStatusBeforeLayout({ });
}

TextBoxTrimmer::TextBoxTrimmer(const RenderBlockFlow& blockContainer, const RenderBlockFlow& lastFormattedLineRoot)
    : m_blockContainer(blockContainer)
{
    adjustTextBoxTrimStatusBeforeLayout(&lastFormattedLineRoot);
}

TextBoxTrimmer::~TextBoxTrimmer()
{
    adjustTextBoxTrimStatusAfterLayout();
}

RenderBlockFlow* TextBoxTrimmer::lastInlineFormattingContextRootForTrimEnd(const RenderBlockFlow& blockContainer)
{
    auto textBoxTrimValue = textBoxTrim(blockContainer);
    auto hasTextBoxTrimEnd = textBoxTrimValue == TextBoxTrim::TrimEnd || textBoxTrimValue == TextBoxTrim::TrimBoth;
    if (!hasTextBoxTrimEnd)
        return { };
    // If the last block container has border/padding end, trimming should not happen.
    if (auto* candidateForLastBlockContainer = lastFormattedLineRoot(blockContainer); candidateForLastBlockContainer && !candidateForLastBlockContainer->borderAndPaddingEnd())
        return candidateForLastBlockContainer;
    return { };
}

void TextBoxTrimmer::adjustTextBoxTrimStatusBeforeLayout(const RenderBlockFlow* lastFormattedLineRoot)
{
    auto textBoxTrimValue = textBoxTrim(*m_blockContainer);
    if (textBoxTrimValue == TextBoxTrim::None)
        return handleTextBoxTrimNoneBeforeLayout();

    auto& layoutContext = m_blockContainer->view().frameView().layoutContext();
    // This block container starts setting up trimming for its subtree.
    // 1. Let's save the current trimming status, merge (and restore after layout).
    // 2. Figure out which side(s) of the content is going to get trimmed.
    m_previousTextBoxTrimStatus = layoutContext.textBoxTrim();
    m_shouldRestoreTextBoxTrimStatus = true;

    auto shouldTrimFirstFormattedLineStart = (textBoxTrimValue == TextBoxTrim::TrimStart || textBoxTrimValue == TextBoxTrim::TrimBoth) || (m_previousTextBoxTrimStatus && m_previousTextBoxTrimStatus->trimFirstFormattedLine);
    auto shouldTrimmingLastFormattedLineEnd = textBoxTrimValue == TextBoxTrim::TrimEnd || textBoxTrimValue == TextBoxTrim::TrimBoth;

    if (shouldTrimmingLastFormattedLineEnd) {
        if (!lastFormattedLineRoot && m_blockContainer->childrenInline()) {
            // Last line end trimming is explicitly set on this inline formatting context. Let's assume last line is part of this block container.
            lastFormattedLineRoot = m_blockContainer.get();
        } else if (lastFormattedLineRoot) {
            // This is the dedicated "last line" layout on the last inline formatting context, where we should not trim the first line
            // unless this IFC includes it too.
            if (shouldTrimFirstFormattedLineStart)
                shouldTrimFirstFormattedLineStart = firstFormattedLineRoot(*m_blockContainer) == lastFormattedLineRoot;
        }
    }
    if (!lastFormattedLineRoot && m_previousTextBoxTrimStatus)
        lastFormattedLineRoot =  m_previousTextBoxTrimStatus->lastFormattedLineRoot.get();

    layoutContext.setTextBoxTrim(LocalFrameViewLayoutContext::TextBoxTrim { shouldTrimFirstFormattedLineStart, lastFormattedLineRoot });
}

void TextBoxTrimmer::adjustTextBoxTrimStatusAfterLayout()
{
    auto& layoutContext = m_blockContainer->view().frameView().layoutContext();
    if (m_shouldRestoreTextBoxTrimStatus)
        return layoutContext.setTextBoxTrim(m_previousTextBoxTrimStatus);

    if (auto textBoxTrim = layoutContext.textBoxTrim(); textBoxTrim && textBoxTrim->trimFirstFormattedLine) {
        // Only the first formatted line is trimmed.
        if (!shouldIgnoreAsFirstLastFormattedLineContainer(*m_blockContainer))
            removeTextBoxTrimStart(layoutContext);
    }
}

void TextBoxTrimmer::handleTextBoxTrimNoneBeforeLayout()
{
    auto& layoutContext = m_blockContainer->view().frameView().layoutContext();
    // This is when the block container does not have text-box-trim set.
    // 1. trimming from ancestors does not get propagated into formatting contexts e.g inside inline-block.
    // 2. border and padding (start) prevent trim start.
    if (m_blockContainer->createsNewFormattingContext()) {
        m_previousTextBoxTrimStatus = layoutContext.textBoxTrim();
        m_shouldRestoreTextBoxTrimStatus = true;
        // Run layout on this subtree with no text-box-trim.
        layoutContext.setTextBoxTrim({ });
        return;
    }

    auto hasTextBoxTrimStart = layoutContext.textBoxTrim() && layoutContext.textBoxTrim()->trimFirstFormattedLine;
    if (hasTextBoxTrimStart && m_blockContainer->borderAndPaddingStart()) {
        // We've got this far with no start trimming and now border/padding prevent trimming.
        removeTextBoxTrimStart(layoutContext);
    }
}

} // namespace WebCore
