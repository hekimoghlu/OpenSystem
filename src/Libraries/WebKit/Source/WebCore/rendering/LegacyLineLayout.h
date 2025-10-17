/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

#include "LegacyInlineFlowBox.h"
#include "LineWidth.h"
#include "RenderLineBoxList.h"
#include "RenderStyleConstants.h"
#include "TrailingObjects.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class BidiContext;
class FloatingObject;
class FloatWithRect;
class LegacyInlineBox;
class LegacyInlineIterator;
class LineInfo;
class LocalFrameViewLayoutContext;
class RenderBlockFlow;
class RenderObject;
class LegacyRootInlineBox;
struct BidiStatus;
struct WordMeasurement;

template <class Run> class BidiRunList;

class LegacyLineLayout {
    WTF_MAKE_TZONE_ALLOCATED(LegacyLineLayout);
public:
    LegacyLineLayout(RenderBlockFlow&);
    ~LegacyLineLayout();

    RenderLineBoxList& lineBoxes() { return m_lineBoxes; }
    const RenderLineBoxList& lineBoxes() const { return m_lineBoxes; }

    LegacyRootInlineBox* legacyRootBox() const { return downcast<LegacyRootInlineBox>(m_lineBoxes.firstLegacyLineBox()); }

    void layoutLineBoxes();

    LegacyRootInlineBox* constructLine(BidiRunList<BidiRun>&, const LineInfo&);
    void addOverflowFromInlineChildren();

    size_t lineCount() const;

    static void appendRunsForObject(BidiRunList<BidiRun>*, int start, int end, RenderObject&, InlineBidiResolver&);

private:
    std::unique_ptr<LegacyRootInlineBox> createRootInlineBox();
    LegacyRootInlineBox* createAndAppendRootInlineBox();
    LegacyInlineBox* createInlineBoxForRenderer(RenderObject*);
    LegacyInlineFlowBox* createLineBoxes(RenderObject*, const LineInfo&, LegacyInlineBox*);
    void removeInlineBox(BidiRun&, const LegacyRootInlineBox&) const;
    void removeEmptyTextBoxesAndUpdateVisualReordering(LegacyRootInlineBox*, BidiRun* firstRun);
    inline BidiRun* handleTrailingSpaces(BidiRunList<BidiRun>& bidiRuns, BidiContext* currentContext);
    LegacyRootInlineBox* createLineBoxesFromBidiRuns(unsigned bidiLevel, BidiRunList<BidiRun>& bidiRuns, const LegacyInlineIterator& end, LineInfo&);
    void layoutRunsAndFloats(bool hasInlineChild);
    void layoutRunsAndFloatsInRange(InlineBidiResolver&);

    const RenderStyle& style() const;
    const LocalFrameViewLayoutContext& layoutContext() const;

    RenderBlockFlow& m_flow;
    RenderLineBoxList m_lineBoxes;
};

};
