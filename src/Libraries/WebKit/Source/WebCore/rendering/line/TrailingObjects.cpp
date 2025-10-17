/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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
#include "TrailingObjects.h"

#include "LegacyInlineIterator.h"
#include "RenderStyleInlines.h"

namespace WebCore {

void TrailingObjects::updateWhitespaceCollapsingTransitionsForTrailingBoxes(LineWhitespaceCollapsingState& lineWhitespaceCollapsingState, const LegacyInlineIterator& lBreak, CollapseFirstSpace collapseFirstSpace)
{
    if (!m_whitespace)
        return;

    // This object is either going to be part of the last transition, or it is going to be the actual endpoint.
    // In both cases we just decrease our pos by 1 level to exclude the space, allowing it to - in effect - collapse into the newline.
    if (lineWhitespaceCollapsingState.numTransitions() % 2) {
        // Find the trailing space object's transition.
        int trailingSpaceTransition = lineWhitespaceCollapsingState.numTransitions() - 1;
        for ( ; trailingSpaceTransition > 0 && lineWhitespaceCollapsingState.transitions()[trailingSpaceTransition].renderer() != m_whitespace; --trailingSpaceTransition) { }
        ASSERT(trailingSpaceTransition >= 0);
        if (collapseFirstSpace == CollapseFirstSpace::Yes)
            lineWhitespaceCollapsingState.decrementTransitionAt(trailingSpaceTransition);

        // Now make sure every single trailingPositionedBox following the trailingSpaceTransition properly stops and starts
        // ignoring spaces.
        size_t currentTransition = trailingSpaceTransition + 1;
        for (size_t i = 0; i < m_boxes.size(); ++i) {
            if (currentTransition >= lineWhitespaceCollapsingState.numTransitions()) {
                // We don't have a transition for this box yet.
                lineWhitespaceCollapsingState.ensureLineBoxInsideIgnoredSpaces(m_boxes[i]);
            } else {
                ASSERT(lineWhitespaceCollapsingState.transitions()[currentTransition].renderer() == &(m_boxes[i].get()));
                ASSERT(lineWhitespaceCollapsingState.transitions()[currentTransition + 1].renderer() == &(m_boxes[i].get()));
            }
            currentTransition += 2;
        }
    } else if (!lBreak.renderer()) {
        ASSERT(m_whitespace->isRenderText());
        ASSERT(collapseFirstSpace == CollapseFirstSpace::Yes);
        // Add a new end transition that stops right at the very end.
        unsigned length = m_whitespace->text().length();
        unsigned pos = length >= 2 ? length - 2 : UINT_MAX;
        LegacyInlineIterator endMid(0, m_whitespace, pos);
        lineWhitespaceCollapsingState.startIgnoringSpaces(endMid);
        for (size_t i = 0; i < m_boxes.size(); ++i)
            lineWhitespaceCollapsingState.ensureLineBoxInsideIgnoredSpaces(m_boxes[i]);
    }
}

}
