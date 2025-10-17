/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#include "RenderTextFragment.h"

#include "RenderBlock.h"
#include "RenderIterator.h"
#include "RenderMultiColumnFlow.h"
#include "RenderStyleInlines.h"
#include "RenderTreeBuilder.h"
#include "Text.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(RenderTextFragment);

RenderTextFragment::RenderTextFragment(Text& textNode, const String& text, int startOffset, int length)
    : RenderText(Type::TextFragment, textNode, text.substring(startOffset, length))
    , m_start(startOffset)
    , m_end(length)
    , m_firstLetter(nullptr)
{
}

RenderTextFragment::RenderTextFragment(Document& document, const String& text, int startOffset, int length)
    : RenderText(Type::TextFragment, document, text.substring(startOffset, length))
    , m_start(startOffset)
    , m_end(length)
    , m_firstLetter(nullptr)
{
}

RenderTextFragment::RenderTextFragment(Document& textNode, const String& text)
    : RenderText(Type::TextFragment, textNode, text)
    , m_start(0)
    , m_end(text.length())
    , m_contentString(text)
    , m_firstLetter(nullptr)
{
}

RenderTextFragment::~RenderTextFragment()
{
    ASSERT(!m_firstLetter);
}

bool RenderTextFragment::canBeSelectionLeaf() const
{
    return textNode() && textNode()->hasEditableStyle();
}

void RenderTextFragment::setTextInternal(const String& newText, bool force)
{
    RenderText::setTextInternal(newText, force);

    m_start = 0;
    m_end = text().length();
    if (!m_firstLetter)
        return;
    if (RenderTreeBuilder::current())
        RenderTreeBuilder::current()->destroy(*m_firstLetter);
    else
        RenderTreeBuilder(*document().renderView()).destroy(*m_firstLetter);
    ASSERT(!m_firstLetter);
    ASSERT(!textNode() || textNode()->renderer() == this);
}

Vector<UChar> RenderTextFragment::previousCharacter() const
{
    if (start()) {
        String original = textNode() ? textNode()->data() : contentString();
        if (!original.isNull() && start() <= original.length()) {
            Vector<UChar> previous;
            previous.append(original[start() - 1]);
            return previous;
        }
    }
    return RenderText::previousCharacter();
}

RenderBlock* RenderTextFragment::blockForAccompanyingFirstLetter()
{
    if (!m_firstLetter)
        return nullptr;
    for (auto& block : ancestorsOfType<RenderBlock>(*m_firstLetter)) {
        if (is<RenderMultiColumnFlow>(block))
            break;
        if (block.style().hasPseudoStyle(PseudoId::FirstLetter) && block.canHaveChildren())
            return &block;
    }
    return nullptr;
}

void RenderTextFragment::setContentString(const String& text)
{
    m_contentString = text;
    setText(text);
}

} // namespace WebCore
