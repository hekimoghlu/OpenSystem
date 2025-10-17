/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 31, 2022.
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

#include "RenderObject.h"

namespace WebCore {

class LegacyInlineFlowBox;
class RenderBlockFlow;

class RenderLineBoxList {
public:
    RenderLineBoxList()
        : m_firstLineBox(nullptr)
        , m_lastLineBox(nullptr)
    {
    }

#ifndef NDEBUG
    ~RenderLineBoxList();
#endif

    LegacyInlineFlowBox* firstLegacyLineBox() const { return m_firstLineBox; }
    LegacyInlineFlowBox* lastLegacyLineBox() const { return m_lastLineBox; }

    void checkConsistency() const;

    void appendLineBox(std::unique_ptr<LegacyInlineFlowBox>);

    void deleteLineBoxTree();
    void deleteLineBoxes();

    void extractLineBox(LegacyInlineFlowBox*);
    void attachLineBox(LegacyInlineFlowBox*);
    void removeLineBox(LegacyInlineFlowBox*);
    
    void dirtyLineBoxes();
    void dirtyLineFromChangedChild(RenderBoxModelObject& parent);
    void shiftLinesBy(LayoutUnit shiftX, LayoutUnit shiftY);

private:
    // For block flows, each box represents the root inline box for a line in the paragraph.
    // For inline flows, each box represents a portion of that inline.
    LegacyInlineFlowBox* m_firstLineBox;
    LegacyInlineFlowBox* m_lastLineBox;
};


#ifdef NDEBUG
inline void RenderLineBoxList::checkConsistency() const
{
}
#endif

} // namespace WebCore
