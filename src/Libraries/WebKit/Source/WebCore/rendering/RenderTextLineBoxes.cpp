/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 11, 2022.
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
#include "RenderTextLineBoxes.h"

#include "LegacyInlineTextBox.h"
#include "LegacyRootInlineBox.h"
#include "RenderBlock.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"
#include "VisiblePosition.h"

namespace WebCore {

RenderTextLineBoxes::RenderTextLineBoxes()
    : m_first(nullptr)
    , m_last(nullptr)
{
}

LegacyInlineTextBox* RenderTextLineBoxes::createAndAppendLineBox(RenderText& renderText)
{
    auto textBox = renderText.createTextBox();
    if (!m_first) {
        m_first = textBox.get();
        m_last = textBox.get();
    } else {
        m_last->setNextTextBox(textBox.get());
        textBox->setPreviousTextBox(m_last);
        m_last = textBox.get();
    }
    return textBox.release();
}

void RenderTextLineBoxes::extract(LegacyInlineTextBox& box)
{
    checkConsistency();

    m_last = box.prevTextBox();
    if (&box == m_first)
        m_first = nullptr;
    if (box.prevTextBox())
        box.prevTextBox()->setNextTextBox(nullptr);
    box.setPreviousTextBox(nullptr);
    for (auto* current = &box; current; current = current->nextTextBox())
        current->setExtracted();

    checkConsistency();
}

void RenderTextLineBoxes::attach(LegacyInlineTextBox& box)
{
    checkConsistency();

    if (m_last) {
        m_last->setNextTextBox(&box);
        box.setPreviousTextBox(m_last);
    } else
        m_first = &box;
    LegacyInlineTextBox* last = nullptr;
    for (auto* current = &box; current; current = current->nextTextBox()) {
        current->setExtracted(false);
        last = current;
    }
    m_last = last;

    checkConsistency();
}

void RenderTextLineBoxes::remove(LegacyInlineTextBox& box)
{
    checkConsistency();

    if (&box == m_first)
        m_first = box.nextTextBox();
    if (&box == m_last)
        m_last = box.prevTextBox();
    if (box.nextTextBox())
        box.nextTextBox()->setPreviousTextBox(box.prevTextBox());
    if (box.prevTextBox())
        box.prevTextBox()->setNextTextBox(box.nextTextBox());

    checkConsistency();
}

void RenderTextLineBoxes::removeAllFromParent(RenderText& renderer)
{
    if (!m_first) {
        if (renderer.parent())
            renderer.parent()->dirtyLineFromChangedChild();
        return;
    }
    for (auto* box = m_first; box; box = box->nextTextBox())
        box->removeFromParent();
}

void RenderTextLineBoxes::deleteAll()
{
    if (!m_first)
        return;
    LegacyInlineTextBox* next;
    for (auto* current = m_first; current; current = next) {
        next = current->nextTextBox();
        delete current;
    }
    m_first = nullptr;
    m_last = nullptr;
}

LegacyInlineTextBox* RenderTextLineBoxes::findNext(int offset, int& position) const
{
    if (!m_first)
        return nullptr;
    // FIXME: This looks buggy. The function is only used for debugging purposes.
    auto current = m_first;
    int currentOffset = current->len();
    while (offset > currentOffset && current->nextTextBox()) {
        current = current->nextTextBox();
        currentOffset = current->start() + current->len();
    }
    // we are now in the correct text run
    position = (offset > currentOffset ? current->len() : current->len() - (currentOffset - offset));
    return current;
}

void RenderTextLineBoxes::dirtyAll()
{
    for (auto* box = m_first; box; box = box->nextTextBox())
        box->dirtyLineBoxes();
}

bool RenderTextLineBoxes::dirtyForTextChange(RenderText& renderer)
{
    dirtyAll();

    if (!m_first && renderer.parent()) {
        renderer.parent()->dirtyLineFromChangedChild();
        return true;
    }
    return m_first;
}

inline void RenderTextLineBoxes::checkConsistency() const
{
#if ASSERT_ENABLED
#ifdef CHECK_CONSISTENCY
    const LegacyInlineTextBox* prev = nullptr;
    for (auto* child = m_first; child; child = child->nextTextBox()) {
        ASSERT(child->renderer() == this);
        ASSERT(child->prevTextBox() == prev);
        prev = child;
    }
    ASSERT(prev == m_last);
#endif
#endif // ASSERT_ENABLED
}

#if ASSERT_ENABLED
RenderTextLineBoxes::~RenderTextLineBoxes()
{
    ASSERT(!m_first);
    ASSERT(!m_last);
}
#endif

#if !ASSERT_WITH_SECURITY_IMPLICATION_DISABLED
void RenderTextLineBoxes::invalidateParentChildLists()
{
    for (auto* box = m_first; box; box = box->nextTextBox())
        box->invalidateParentChildList();
}
#endif

}
