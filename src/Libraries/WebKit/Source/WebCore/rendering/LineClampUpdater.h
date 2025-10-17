/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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

#include "RenderLayoutState.h"
#include "RenderView.h"
#include <wtf/CheckedPtr.h>

namespace WebCore {

class LineClampUpdater {
public:
    LineClampUpdater(const RenderBlockFlow& blockContainer);
    ~LineClampUpdater();

private:
    CheckedPtr<const RenderBlockFlow> m_blockContainer;
    std::optional<RenderLayoutState::LineClamp> m_previousLineClamp { };
    std::optional<RenderLayoutState::LegacyLineClamp> m_skippedLegacyLineClampToRestore { };
};

inline LineClampUpdater::LineClampUpdater(const RenderBlockFlow& blockContainer)
    : m_blockContainer(blockContainer)
{
    auto* layoutState = m_blockContainer->view().frameView().layoutContext().layoutState();
    if (!layoutState)
        return;

    m_previousLineClamp = layoutState->lineClamp();
    if (blockContainer.isFieldset()) {
        layoutState->setLineClamp({ });

        m_skippedLegacyLineClampToRestore = layoutState->legacyLineClamp();
        layoutState->setLegacyLineClamp({ });
        return;
    }

    auto maximumLinesForBlockContainer = m_blockContainer->style().maxLines();
    if (maximumLinesForBlockContainer) {
        // New, top level line clamp.
        layoutState->setLineClamp(RenderLayoutState::LineClamp { maximumLinesForBlockContainer, m_blockContainer->style().overflowContinue() == OverflowContinue::Discard });
        return;
    }

    if (m_previousLineClamp) {
        // Propagated line clamp.
        if (blockContainer.establishesIndependentFormattingContext()) {
            // Contents of descendants that establish independent formatting contexts are skipped over while counting line boxes.
            layoutState->setLineClamp({ });
            return;
        }
        auto effectiveShouldDiscard = m_previousLineClamp->shouldDiscardOverflow  || m_blockContainer->style().overflowContinue() == OverflowContinue::Discard;
        layoutState->setLineClamp(RenderLayoutState::LineClamp { m_previousLineClamp->maximumLines, effectiveShouldDiscard });
        return;
    }
}

inline LineClampUpdater::~LineClampUpdater()
{
    auto* layoutState = m_blockContainer->view().frameView().layoutContext().layoutState();
    if (!layoutState)
        return;

    if (m_skippedLegacyLineClampToRestore)
        layoutState->setLegacyLineClamp(m_skippedLegacyLineClampToRestore);

    if (!m_previousLineClamp) {
        layoutState->setLineClamp({ });
        return;
    }

    layoutState->setLineClamp(RenderLayoutState::LineClamp { m_previousLineClamp->maximumLines - (m_blockContainer->childrenInline() ? m_blockContainer->lineCount() : 0), m_previousLineClamp->shouldDiscardOverflow });
}

}
