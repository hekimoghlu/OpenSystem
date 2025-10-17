/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 25, 2022.
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
#include "LayoutRepainter.h"

#include "RenderElement.h"
#include "RenderLayerModelObject.h"

namespace WebCore {

LayoutRepainter::LayoutRepainter(RenderElement& renderer, std::optional<CheckForRepaint> checkForRepaintOverride, std::optional<ShouldAlwaysIssueFullRepaint> shouldAlwaysIssueFullRepaint, RepaintOutlineBounds repaintOutlineBounds)
    : m_renderer(renderer)
    , m_checkForRepaint(checkForRepaintOverride ? *checkForRepaintOverride == CheckForRepaint::Yes : m_renderer->checkForRepaintDuringLayout())
    , m_forceFullRepaint(shouldAlwaysIssueFullRepaint && *shouldAlwaysIssueFullRepaint == ShouldAlwaysIssueFullRepaint::Yes)
    , m_repaintOutlineBounds(repaintOutlineBounds)
{
    if (!m_checkForRepaint)
        return;

    m_repaintContainer = m_renderer->containerForRepaint().renderer.get();
    m_oldRects = m_renderer->rectsForRepaintingAfterLayout(m_repaintContainer, m_repaintOutlineBounds);
}

bool LayoutRepainter::repaintAfterLayout()
{
    if (!m_checkForRepaint)
        return false;

    auto requiresFullRepaint = m_forceFullRepaint || m_renderer->selfNeedsLayout() ? RequiresFullRepaint::Yes : RequiresFullRepaint::No;
    // Outline bounds are not used if we're doing a full repaint.
    auto newRects = m_renderer->rectsForRepaintingAfterLayout(m_repaintContainer, (requiresFullRepaint == RequiresFullRepaint::Yes) ? RepaintOutlineBounds::No : m_repaintOutlineBounds);
    return m_renderer->repaintAfterLayoutIfNeeded(m_repaintContainer, requiresFullRepaint, m_oldRects, newRects);
}

} // namespace WebCore
