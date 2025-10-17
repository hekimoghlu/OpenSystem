/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 8, 2021.
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
#include "ImageOverlayController.h"

#include "Chrome.h"
#include "ChromeClient.h"
#include "Document.h"
#include "Editor.h"
#include "FrameSelection.h"
#include "GraphicsContext.h"
#include "HTMLElement.h"
#include "ImageOverlay.h"
#include "IntRect.h"
#include "LayoutRect.h"
#include "LocalFrame.h"
#include "Page.h"
#include "PageOverlayController.h"
#include "PlatformMouseEvent.h"
#include "RenderElement.h"
#include "RenderStyleInlines.h"
#include "SimpleRange.h"
#include "VisiblePosition.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageOverlayController);

class FloatQuad;

ImageOverlayController::ImageOverlayController(Page& page)
    : m_page(page)
{
}

void ImageOverlayController::selectionQuadsDidChange(LocalFrame& frame, const Vector<FloatQuad>& quads)
{
    if (!m_page || !protectedPage()->chrome().client().needsImageOverlayControllerForSelectionPainting())
        return;

    if (frame.editor().ignoreSelectionChanges() || frame.editor().isGettingDictionaryPopupInfo())
        return;

    m_hostElementForSelection = nullptr;
    m_selectionQuads.clear();
    m_selectionBackgroundColor = Color::transparentBlack;
    m_selectionClipRect = { };

    auto overlayHost = ([&] () -> RefPtr<HTMLElement> {
        auto selectedRange = frame.selection().selection().range();
        if (!selectedRange)
            return nullptr;

        if (!ImageOverlay::isInsideOverlay(*selectedRange))
            return nullptr;

        if (RefPtr host = selectedRange->startContainer().shadowHost(); is<HTMLElement>(host))
            return static_pointer_cast<HTMLElement>(WTFMove(host));

        return nullptr;
    })();

    if (!overlayHost) {
        uninstallPageOverlayIfNeeded();
        return;
    }

    auto overlayHostRenderer = overlayHost->renderer();
    if (!overlayHostRenderer) {
        uninstallPageOverlayIfNeeded();
        return;
    }

    if (!shouldUsePageOverlayToPaintSelection(*overlayHostRenderer)) {
        uninstallPageOverlayIfNeeded();
        return;
    }

    m_hostElementForSelection = *overlayHost;
    m_selectionQuads = quads;
    m_selectionBackgroundColor = overlayHostRenderer->selectionBackgroundColor();
    m_selectionClipRect = overlayHostRenderer->absoluteBoundingBoxRect();

    installPageOverlayIfNeeded().setNeedsDisplay();
}

bool ImageOverlayController::shouldUsePageOverlayToPaintSelection(const RenderElement& renderer)
{
    // If the selection is already painted (with nonzero opacity) in the overlay host's renderer,
    // then we don't need to fall back to a page overlay to paint the selection.
    return renderer.style().opacity() <= 0.01;
}

void ImageOverlayController::documentDetached(const Document& document)
{
    if (m_hostElementForSelection && &document == &m_hostElementForSelection->document())
        m_hostElementForSelection = nullptr;

#if PLATFORM(MAC)
    if (m_hostElementForDataDetectors && &document == &m_hostElementForDataDetectors->document())
        m_hostElementForDataDetectors = nullptr;
#endif

    uninstallPageOverlayIfNeeded();
}

PageOverlay& ImageOverlayController::installPageOverlayIfNeeded()
{
    if (m_overlay)
        return *m_overlay;

    m_overlay = PageOverlay::create(*this, PageOverlay::OverlayType::Document);
    protectedPage()->pageOverlayController().installPageOverlay(*protectedOverlay(), PageOverlay::FadeMode::DoNotFade);
    return *m_overlay;
}

void ImageOverlayController::uninstallPageOverlay()
{
    m_hostElementForSelection = nullptr;
    m_selectionQuads.clear();
    m_selectionBackgroundColor = Color::transparentBlack;
    m_selectionClipRect = { };

#if PLATFORM(MAC)
    clearDataDetectorHighlights();
#endif

    auto overlayToUninstall = std::exchange(m_overlay, nullptr);
    if (!m_page || !overlayToUninstall)
        return;

    protectedPage()->pageOverlayController().uninstallPageOverlay(*overlayToUninstall, PageOverlay::FadeMode::DoNotFade);
}

RefPtr<Page> ImageOverlayController::protectedPage() const
{
    return m_page.get();
}

void ImageOverlayController::uninstallPageOverlayIfNeeded()
{
    if (m_hostElementForSelection)
        return;

#if PLATFORM(MAC)
    if (m_hostElementForDataDetectors)
        return;
#endif

    uninstallPageOverlay();
}

void ImageOverlayController::willMoveToPage(PageOverlay&, Page* page)
{
    if (!page)
        uninstallPageOverlay();
}

void ImageOverlayController::drawRect(PageOverlay& pageOverlay, GraphicsContext& context, const IntRect& dirtyRect)
{
    if (&pageOverlay != m_overlay) {
        ASSERT_NOT_REACHED();
        return;
    }

    GraphicsContextStateSaver stateSaver(context);
    context.clearRect(dirtyRect);

    if (m_selectionQuads.isEmpty())
        return;

    Path coalescedSelectionPath;
    for (auto& quad : m_selectionQuads) {
        coalescedSelectionPath.moveTo(quad.p1());
        coalescedSelectionPath.addLineTo(quad.p2());
        coalescedSelectionPath.addLineTo(quad.p3());
        coalescedSelectionPath.addLineTo(quad.p4());
        coalescedSelectionPath.addLineTo(quad.p1());
        coalescedSelectionPath.closeSubpath();
    }

    context.setFillColor(m_selectionBackgroundColor);
    context.clip(m_selectionClipRect);
    context.fillPath(coalescedSelectionPath);
}

#if !PLATFORM(MAC)

bool ImageOverlayController::platformHandleMouseEvent(const PlatformMouseEvent&)
{
    return false;
}

void ImageOverlayController::elementUnderMouseDidChange(LocalFrame&, Element*)
{
}

#if ENABLE(DATA_DETECTION)

void ImageOverlayController::textRecognitionResultsChanged(HTMLElement&)
{
}

bool ImageOverlayController::hasActiveDataDetectorHighlightForTesting() const
{
    return false;
}

#endif // ENABLE(DATA_DETECTION)

#endif // !PLATFORM(MAC)

} // namespace WebCore
