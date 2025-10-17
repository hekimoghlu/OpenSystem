/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 4, 2024.
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
#include "RenderSelectionGeometry.h"

#include "RenderText.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderSelectionGeometryBase);
WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderSelectionGeometry);
WTF_MAKE_TZONE_ALLOCATED_IMPL(RenderBlockSelectionGeometry);

RenderSelectionGeometryBase::RenderSelectionGeometryBase(RenderObject& renderer)
    : m_renderer(renderer)
    , m_repaintContainer(renderer.containerForRepaint().renderer.get())
    , m_state(renderer.selectionState())
{
}

void RenderSelectionGeometryBase::repaintRectangle(const LayoutRect& repaintRect)
{
    m_renderer.repaintUsingContainer(m_repaintContainer, enclosingIntRect(repaintRect));
}

RenderSelectionGeometry::RenderSelectionGeometry(RenderObject& renderer, bool clipToVisibleContent)
    : RenderSelectionGeometryBase(renderer)
{
    if (renderer.canUpdateSelectionOnRootLineBoxes()) {
        if (CheckedPtr textRenderer = dynamicDowncast<RenderText>(renderer))
            m_rect = textRenderer->collectSelectionGeometriesForLineBoxes(m_repaintContainer, clipToVisibleContent, m_collectedSelectionQuads);
        else
            m_rect = renderer.selectionRectForRepaint(m_repaintContainer, clipToVisibleContent);
    }
}

void RenderSelectionGeometry::repaint()
{
    repaintRectangle(m_rect);
}

RenderBlockSelectionGeometry::RenderBlockSelectionGeometry(RenderBlock& renderer)
    : RenderSelectionGeometryBase(renderer)
    , m_rects(renderer.canUpdateSelectionOnRootLineBoxes() ? renderer.selectionGapRectsForRepaint(m_repaintContainer) : GapRects())
{
}

void RenderBlockSelectionGeometry::repaint()
{
    repaintRectangle(m_rects);
}

} // namespace WebCore
