/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#include "LegacyRootInlineBox.h"

#include "BidiResolver.h"
#include "CSSLineBoxContainValue.h"
#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "GraphicsContext.h"
#include "HitTestResult.h"
#include "LegacyInlineTextBox.h"
#include "LocalFrame.h"
#include "LogicalSelectionOffsetCaches.h"
#include "PaintInfo.h"
#include "RenderBlockInlines.h"
#include "RenderBoxInlines.h"
#include "RenderFragmentedFlow.h"
#include "RenderInline.h"
#include "RenderLayoutState.h"
#include "RenderView.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(LegacyRootInlineBox);

struct SameSizeAsLegacyRootInlineBox : LegacyInlineFlowBox, CanMakeWeakPtr<LegacyRootInlineBox>, CanMakeCheckedPtr<SameSizeAsLegacyRootInlineBox> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(SameSizeAsLegacyRootInlineBox);

    int layoutUnits[4];
};

static_assert(sizeof(LegacyRootInlineBox) == sizeof(SameSizeAsLegacyRootInlineBox), "LegacyRootInlineBox should stay small");
#if !ASSERT_ENABLED
static_assert(sizeof(SingleThreadWeakPtr<RenderObject>) == sizeof(void*), "WeakPtr should be same size as raw pointer");
#endif

LegacyRootInlineBox::LegacyRootInlineBox(RenderBlockFlow& block)
    : LegacyInlineFlowBox(block)
{
    setIsHorizontal(block.isHorizontalWritingMode());
}

LegacyRootInlineBox::~LegacyRootInlineBox()
{
}

LayoutUnit LegacyRootInlineBox::baselinePosition(FontBaseline baselineType) const
{
    return renderer().baselinePosition(baselineType, isFirstLine(), isHorizontal() ? HorizontalLine : VerticalLine, PositionOfInteriorLineBoxes);
}

LayoutUnit LegacyRootInlineBox::lineHeight() const
{
    return renderer().lineHeight(isFirstLine(), isHorizontal() ? HorizontalLine : VerticalLine, PositionOfInteriorLineBoxes);
}

void LegacyRootInlineBox::adjustPosition(float dx, float dy)
{
    LegacyInlineFlowBox::adjustPosition(dx, dy);
    LayoutUnit blockDirectionDelta { isHorizontal() ? dy : dx }; // The block direction delta is a LayoutUnit.
    m_lineTop += blockDirectionDelta;
    m_lineBottom += blockDirectionDelta;
    m_lineBoxTop += blockDirectionDelta;
    m_lineBoxBottom += blockDirectionDelta;
}

RenderObject::HighlightState LegacyRootInlineBox::selectionState() const
{
    // Walk over all of the selected boxes.
    RenderObject::HighlightState state = RenderObject::HighlightState::None;
    for (auto* box = firstLeafDescendant(); box; box = box->nextLeafOnLine()) {
        RenderObject::HighlightState boxState = box->selectionState();
        if ((boxState == RenderObject::HighlightState::Start && state == RenderObject::HighlightState::End)
            || (boxState == RenderObject::HighlightState::End && state == RenderObject::HighlightState::Start))
            state = RenderObject::HighlightState::Both;
        else if (state == RenderObject::HighlightState::None || ((boxState == RenderObject::HighlightState::Start || boxState == RenderObject::HighlightState::End)
            && (state == RenderObject::HighlightState::None || state == RenderObject::HighlightState::Inside)))
            state = boxState;
        else if (boxState == RenderObject::HighlightState::None && state == RenderObject::HighlightState::Start) {
            // We are past the end of the selection.
            state = RenderObject::HighlightState::Both;
        }
        if (state == RenderObject::HighlightState::Both)
            break;
    }

    return state;
}

const LegacyInlineBox* LegacyRootInlineBox::firstSelectedBox() const
{
    for (auto* box = firstLeafDescendant(); box; box = box->nextLeafOnLine()) {
        if (box->selectionState() != RenderObject::HighlightState::None)
            return box;
    }
    return nullptr;
}

const LegacyInlineBox* LegacyRootInlineBox::lastSelectedBox() const
{
    for (auto* box = lastLeafDescendant(); box; box = box->previousLeafOnLine()) {
        if (box->selectionState() != RenderObject::HighlightState::None)
            return box;
    }
    return nullptr;
}

LayoutUnit LegacyRootInlineBox::selectionTop() const
{
    LayoutUnit selectionTop = m_lineTop;

    if (renderer().style().writingMode().isLineInverted())
        return selectionTop;

    LayoutUnit prevBottom;
    if (auto* previousBox = prevRootBox())
        prevBottom = previousBox->selectionBottom();
    else
        prevBottom = selectionTop;

    return prevBottom;
}

LayoutUnit LegacyRootInlineBox::selectionTopAdjustedForPrecedingBlock() const
{
    return blockFlow().adjustEnclosingTopForPrecedingBlock(selectionTop());
}

LayoutUnit LegacyRootInlineBox::selectionBottom() const
{
    LayoutUnit selectionBottom = m_lineBottom;

    if (!renderer().style().writingMode().isLineInverted() || !nextRootBox())
        return selectionBottom;

    return nextRootBox()->selectionTop();
}

RenderBlockFlow& LegacyRootInlineBox::blockFlow() const
{
    return downcast<RenderBlockFlow>(renderer());
}

void LegacyRootInlineBox::removeLineBoxFromRenderObject()
{
    // Null if we are destroying LegacyLineLayout.
    if (auto* legacyLineLayout = blockFlow().svgTextLayout())
        legacyLineLayout->lineBoxes().removeLineBox(this);
}

void LegacyRootInlineBox::extractLineBoxFromRenderObject()
{
    blockFlow().svgTextLayout()->lineBoxes().extractLineBox(this);
}

void LegacyRootInlineBox::attachLineBoxToRenderObject()
{
    blockFlow().svgTextLayout()->lineBoxes().attachLineBox(this);
}

LayoutUnit LegacyRootInlineBox::lineBoxWidth() const
{
    return blockFlow().availableLogicalWidthForLine(lineBoxTop(), lineBoxHeight());
}

#if ENABLE(TREE_DEBUGGING)

void LegacyRootInlineBox::outputLineBox(WTF::TextStream& stream, bool mark, int depth) const
{
    stream << "-------- --";
    int printedCharacters = 0;
    while (++printedCharacters <= depth * 2)
        stream << " ";
    stream << "Line: (top: " << lineTop() << " bottom: " << lineBottom() << ") with leading (top: " << lineBoxTop() << " bottom: " << lineBoxBottom() << ")";
    stream.nextLine();
    LegacyInlineBox::outputLineBox(stream, mark, depth);
}

ASCIILiteral LegacyRootInlineBox::boxName() const
{
    return "RootInlineBox"_s;
}
#endif

} // namespace WebCore
