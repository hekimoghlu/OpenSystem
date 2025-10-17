/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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
#import "config.h"
#import "FrameSelection.h"

#import "AXObjectCache.h"
#import "Chrome.h"
#import "ChromeClient.h"
#import "DocumentInlines.h"
#import "LocalFrame.h"
#import "RenderView.h"

namespace WebCore {

void FrameSelection::notifyAccessibilityForSelectionChange(const AXTextStateChangeIntent& intent)
{
    if (!AXObjectCache::accessibilityEnabled())
        return;

    if (CheckedPtr cache = m_document->existingAXObjectCache()) {
        if (m_selection.start().isNotNull() && m_selection.end().isNotNull())
            cache->postTextStateChangeNotification(m_selection.start(), intent, m_selection);
        else {
            // The selection was cleared, so use `onSelectedTextChanged` with an empty selection range to update the isolated tree.
            // FIXME: https://bugs.webkit.org/show_bug.cgi?id=279524: In the future, we should consider actually doing
            // `postTextStateChangeNotification`, since right now, VoiceOver does not announce when text becomes unselected
            // (because we don't post a notification), unless new text becomes selected at the same time. However, even if we
            // did post this notification, changes would be needed in VoiceOver too, so just do onSelectedTextChanged for now.
            // Handling selection removals should also not be platform-specific (as it is here with ENABLE(ACCESSIBILITY_ISOLATED_TREE)).
#if ENABLE(ACCESSIBILITY_ISOLATED_TREE)
            cache->onSelectedTextChanged(m_selection);
#endif
        }
    }

#if !PLATFORM(IOS_FAMILY)
    // if zoom feature is enabled, insertion point changes should update the zoom
    if (!m_selection.isCaret())
        return;

    auto* renderView = m_document->renderView();
    if (!renderView)
        return;
    auto* frameView = m_document->view();
    if (!frameView)
        return;

    IntRect selectionRect = absoluteCaretBounds();
    IntRect viewRect = snappedIntRect(renderView->viewRect());

    auto remoteFrameOffset = frameView->frame().loader().client().accessibilityRemoteFrameOffset();
    selectionRect.moveBy({ remoteFrameOffset });
    viewRect.moveBy({ remoteFrameOffset });

    selectionRect = frameView->contentsToScreen(selectionRect);
    viewRect = frameView->contentsToScreen(viewRect);

    if (!m_document->page())
        return;
    
    m_document->page()->chrome().client().changeUniversalAccessZoomFocus(viewRect, selectionRect);
#endif // !PLATFORM(IOS_FAMILY)
}

} // namespace WebCore
