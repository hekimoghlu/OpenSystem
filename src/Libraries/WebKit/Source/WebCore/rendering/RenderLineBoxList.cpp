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
#include "config.h"
#include "RenderLineBoxList.h"

#include "HitTestResult.h"
#include "LegacyInlineTextBox.h"
#include "LegacyRootInlineBox.h"
#include "PaintInfo.h"
#include "RenderBlockFlow.h"
#include "RenderInline.h"
#include "RenderLineBreak.h"
#include "RenderSVGInline.h"
#include "RenderStyleInlines.h"
#include "RenderView.h"

namespace WebCore {

#ifndef NDEBUG
RenderLineBoxList::~RenderLineBoxList()
{
    ASSERT(!m_firstLineBox);
    ASSERT(!m_lastLineBox);
}
#endif

void RenderLineBoxList::appendLineBox(std::unique_ptr<LegacyInlineFlowBox> box)
{
    checkConsistency();

    LegacyInlineFlowBox* boxPtr = box.release();

    if (!m_firstLineBox) {
        m_firstLineBox = boxPtr;
        m_lastLineBox = boxPtr;
    } else {
        m_lastLineBox->setNextLineBox(boxPtr);
        boxPtr->setPreviousLineBox(m_lastLineBox);
        m_lastLineBox = boxPtr;
    }

    checkConsistency();
}

void RenderLineBoxList::deleteLineBoxTree()
{
    LegacyInlineFlowBox* line = m_firstLineBox;
    LegacyInlineFlowBox* nextLine;
    while (line) {
        nextLine = line->nextLineBox();
        line->deleteLine();
        line = nextLine;
    }
    m_firstLineBox = m_lastLineBox = nullptr;
}

void RenderLineBoxList::extractLineBox(LegacyInlineFlowBox* box)
{
    checkConsistency();
    
    m_lastLineBox = box->prevLineBox();
    if (box == m_firstLineBox)
        m_firstLineBox = 0;
    if (box->prevLineBox())
        box->prevLineBox()->setNextLineBox(nullptr);
    box->setPreviousLineBox(nullptr);
    for (auto* curr = box; curr; curr = curr->nextLineBox())
        curr->setExtracted();

    checkConsistency();
}

void RenderLineBoxList::attachLineBox(LegacyInlineFlowBox* box)
{
    checkConsistency();

    if (m_lastLineBox) {
        m_lastLineBox->setNextLineBox(box);
        box->setPreviousLineBox(m_lastLineBox);
    } else
        m_firstLineBox = box;
    LegacyInlineFlowBox* last = box;
    for (auto* curr = box; curr; curr = curr->nextLineBox()) {
        curr->setExtracted(false);
        last = curr;
    }
    m_lastLineBox = last;

    checkConsistency();
}

void RenderLineBoxList::removeLineBox(LegacyInlineFlowBox* box)
{
    checkConsistency();

    if (box == m_firstLineBox)
        m_firstLineBox = box->nextLineBox();
    if (box == m_lastLineBox)
        m_lastLineBox = box->prevLineBox();
    if (box->nextLineBox())
        box->nextLineBox()->setPreviousLineBox(box->prevLineBox());
    if (box->prevLineBox())
        box->prevLineBox()->setNextLineBox(box->nextLineBox());

    checkConsistency();
}

void RenderLineBoxList::deleteLineBoxes()
{
    if (m_firstLineBox) {
        LegacyInlineFlowBox* next;
        for (auto* curr = m_firstLineBox; curr; curr = next) {
            next = curr->nextLineBox();
            delete curr;
        }
        m_firstLineBox = nullptr;
        m_lastLineBox = nullptr;
    }
}

void RenderLineBoxList::dirtyLineBoxes()
{
    for (auto* curr = firstLegacyLineBox(); curr; curr = curr->nextLineBox())
        curr->dirtyLineBoxes();
}

void RenderLineBoxList::shiftLinesBy(LayoutUnit shiftX, LayoutUnit shiftY)
{
    for (auto* box = firstLegacyLineBox(); box; box = box->nextLineBox())
        box->adjustPosition(shiftX, shiftY);
}

void RenderLineBoxList::dirtyLineFromChangedChild(RenderBoxModelObject& container)
{
    ASSERT(is<RenderInline>(container) || is<RenderBlockFlow>(container));
    if (!container.isSVGRenderer())
        return;

    if (!container.parent() || (is<RenderBlockFlow>(container) && container.selfNeedsLayout()))
        return;

    auto* inlineContainer = dynamicDowncast<RenderSVGInline>(container);
    if (auto* lineBox = inlineContainer ? inlineContainer->firstLegacyInlineBox() : firstLegacyLineBox()) {
        lineBox->root().markDirty();
        return;
    }
    // For an empty inline, propagate the check up to our parent.
    if (inlineContainer && inlineContainer->everHadLayout()) {
        auto* parent = inlineContainer->parent();
        parent->dirtyLineFromChangedChild();
        parent->setNeedsLayout();
    }
}

#ifndef NDEBUG

void RenderLineBoxList::checkConsistency() const
{
#ifdef CHECK_CONSISTENCY
    const LegacyInlineFlowBox* prev = nullptr;
    for (const LegacyInlineFlowBox* child = m_firstLineBox; child != nullptr; child = child->nextLineBox()) {
        ASSERT(child->prevLineBox() == prev);
        prev = child;
    }
    ASSERT(prev == m_lastLineBox);
#endif
}

#endif

}
