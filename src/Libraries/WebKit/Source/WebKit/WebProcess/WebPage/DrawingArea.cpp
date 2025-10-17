/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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
#include "DrawingArea.h"

#include "DrawingAreaMessages.h"
#include "Logging.h"
#include "WebPage.h"
#include "WebPageCreationParameters.h"
#include "WebProcess.h"
#include <WebCore/DisplayRefreshMonitor.h>
#include <WebCore/LocalFrameView.h>
#include <WebCore/RenderView.h>
#include <WebCore/ScrollView.h>
#include <WebCore/TiledBacking.h>
#include <WebCore/TransformationMatrix.h>
#include <wtf/TZoneMallocInlines.h>

// Subclasses
#if PLATFORM(COCOA)
#include "RemoteLayerTreeDrawingAreaMac.h"
#include "TiledCoreAnimationDrawingArea.h"
#elif USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)
#include "DrawingAreaCoordinatedGraphics.h"
#endif
#if USE(GRAPHICS_LAYER_WC)
#include "DrawingAreaWC.h"
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(DrawingArea);

RefPtr<DrawingArea> DrawingArea::create(WebPage& webPage, const WebPageCreationParameters& parameters)
{
#if PLATFORM(MAC)
    SandboxExtension::consumePermanently(parameters.renderServerMachExtensionHandle);
#endif

    switch (parameters.drawingAreaType) {
#if PLATFORM(COCOA)
#if !PLATFORM(IOS_FAMILY)
    case DrawingAreaType::TiledCoreAnimation:
        return TiledCoreAnimationDrawingArea::create(webPage, parameters);
#endif
    case DrawingAreaType::RemoteLayerTree:
#if PLATFORM(MAC)
        return RemoteLayerTreeDrawingAreaMac::create(webPage, parameters);
#else
        return RemoteLayerTreeDrawingArea::create(webPage, parameters);
#endif
#elif USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)
    case DrawingAreaType::CoordinatedGraphics:
        return DrawingAreaCoordinatedGraphics::create(webPage, parameters);
#endif
#if USE(GRAPHICS_LAYER_WC)
    case DrawingAreaType::WC:
        return DrawingAreaWC::create(webPage, parameters);
#endif
    }

    return nullptr;
}

DrawingArea::DrawingArea(DrawingAreaType type, DrawingAreaIdentifier identifier, WebPage& webPage)
    : m_type(type)
    , m_identifier(identifier)
    , m_webPage(webPage)
{
    WebProcess::singleton().addMessageReceiver(Messages::DrawingArea::messageReceiverName(), m_identifier, *this);
}

DrawingArea::~DrawingArea()
{
    removeMessageReceiverIfNeeded();
}

DelegatedScrollingMode DrawingArea::delegatedScrollingMode() const
{
    return DelegatedScrollingMode::NotDelegated;
}

void DrawingArea::dispatchAfterEnsuringUpdatedScrollPosition(WTF::Function<void ()>&& function)
{
    // Scroll position updates are synchronous by default so we can just call the function right away here.
    function();
}

void DrawingArea::tryMarkLayersVolatile(CompletionHandler<void(bool)>&& completionFunction)
{
    completionFunction(true);
}

void DrawingArea::removeMessageReceiverIfNeeded()
{
    if (m_hasRemovedMessageReceiver)
        return;
    m_hasRemovedMessageReceiver = true;
    WebProcess::singleton().removeMessageReceiver(Messages::DrawingArea::messageReceiverName(), m_identifier);
}

RefPtr<WebCore::DisplayRefreshMonitor> DrawingArea::createDisplayRefreshMonitor(WebCore::PlatformDisplayID)
{
    return nullptr;
}

void DrawingArea::willStartRenderingUpdateDisplay()
{
    Ref { m_webPage.get() }->willStartRenderingUpdateDisplay();
}

void DrawingArea::didCompleteRenderingUpdateDisplay()
{
    Ref { m_webPage.get() }->didCompleteRenderingUpdateDisplay();
}

void DrawingArea::didCompleteRenderingFrame()
{
    Ref { m_webPage.get() }->didCompleteRenderingFrame();
}

bool DrawingArea::supportsGPUProcessRendering(DrawingAreaType type)
{
    switch (type) {
#if PLATFORM(COCOA)
#if !PLATFORM(IOS_FAMILY)
    case DrawingAreaType::TiledCoreAnimation:
        return false;
#endif
    case DrawingAreaType::RemoteLayerTree:
        return true;
#elif USE(COORDINATED_GRAPHICS) || USE(TEXTURE_MAPPER)
    case DrawingAreaType::CoordinatedGraphics:
        return false;
#endif
#if USE(GRAPHICS_LAYER_WC)
    case DrawingAreaType::WC:
        return true;
#endif
    default:
        return false;
    }
}

WebCore::TiledBacking* DrawingArea::mainFrameTiledBacking() const
{
    RefPtr frameView = m_webPage->localMainFrameView();
    return frameView ? frameView->tiledBacking() : nullptr;
}

void DrawingArea::prepopulateRectForZoom(double scale, WebCore::FloatPoint origin)
{
    Ref webPage = m_webPage.get();
    double currentPageScale = webPage->totalScaleFactor();
    auto* frameView = webPage->localMainFrameView();
    if (!frameView)
        return;

    FloatRect tileCoverageRect = frameView->visibleContentRectIncludingScrollbars();
    tileCoverageRect.moveBy(-origin);
    tileCoverageRect.scale(currentPageScale / scale);

    if (auto* tiledBacking = mainFrameTiledBacking())
        tiledBacking->prepopulateRect(tileCoverageRect);
}

void DrawingArea::scaleViewToFitDocumentIfNeeded()
{
    const int maximumDocumentWidthForScaling = 1440;
    const float minimumViewScale = 0.1;

    if (!m_shouldScaleViewToFitDocument)
        return;

    LOG(Resize, "DrawingArea %p scaleViewToFitDocumentIfNeeded", this);
    Ref webPage = m_webPage.get();
    webPage->layoutIfNeeded();

    if (!webPage->localMainFrameView() || !webPage->localMainFrameView()->renderView())
        return;

    int viewWidth = webPage->size().width();
    int documentWidth = webPage->localMainFrameView()->renderView()->unscaledDocumentRect().width();

    bool documentWidthChanged = m_lastDocumentSizeForScaleToFit.width() != documentWidth;
    bool viewWidthChanged = m_lastViewSizeForScaleToFit.width() != viewWidth;

    LOG(Resize, "  documentWidthChanged=%d, viewWidthChanged=%d", documentWidthChanged, viewWidthChanged);

    if (!documentWidthChanged && !viewWidthChanged)
        return;

    // The view is now bigger than the document, so we'll re-evaluate whether we have to scale.
    if (m_isScalingViewToFitDocument && viewWidth >= m_lastDocumentSizeForScaleToFit.width())
        m_isScalingViewToFitDocument = false;

    // Our current understanding of the document width is still up to date, and we're in scaling mode.
    // Update the viewScale without doing an extra layout to re-determine the document width.
    if (m_isScalingViewToFitDocument) {
        if (!documentWidthChanged) {
            m_lastViewSizeForScaleToFit = webPage->size();
            float viewScale = (float)viewWidth / (float)m_lastDocumentSizeForScaleToFit.width();
            if (viewScale < minimumViewScale) {
                viewScale = minimumViewScale;
                documentWidth = std::ceil(viewWidth / viewScale);
            }
            IntSize fixedLayoutSize(documentWidth, std::ceil((webPage->size().height() - webPage->corePage()->topContentInset()) / viewScale));
            webPage->setFixedLayoutSize(fixedLayoutSize);
            webPage->scaleView(viewScale);

            LOG(Resize, "  using fixed layout at %dx%d. document width %d unchanged, scaled to %.4f to fit view width %d", fixedLayoutSize.width(), fixedLayoutSize.height(), documentWidth, viewScale, viewWidth);
            return;
        }
    
        IntSize fixedLayoutSize = webPage->fixedLayoutSize();
        if (documentWidth > fixedLayoutSize.width()) {
            LOG(Resize, "  page laid out wider than fixed layout width. Not attempting to re-scale");
            return;
        }
    }

    LOG(Resize, "  doing unconstrained layout");

    // Lay out at the view size.
    webPage->setUseFixedLayout(false);
    webPage->layoutIfNeeded();

    if (!webPage->localMainFrameView() || !webPage->localMainFrameView()->renderView())
        return;

    IntSize documentSize = webPage->localMainFrameView()->renderView()->unscaledDocumentRect().size();
    m_lastViewSizeForScaleToFit = webPage->size();
    m_lastDocumentSizeForScaleToFit = documentSize;

    documentWidth = documentSize.width();

    float viewScale = 1;

    LOG(Resize, "  unscaled document size %dx%d. need to scale down: %d", documentSize.width(), documentSize.height(), documentWidth && documentWidth < maximumDocumentWidthForScaling && viewWidth < documentWidth);

    // Avoid scaling down documents that don't fit in a certain width, to allow
    // sites that want horizontal scrollbars to continue to have them.
    if (documentWidth && documentWidth < maximumDocumentWidthForScaling && viewWidth < documentWidth) {
        // If the document doesn't fit in the view, scale it down but lay out at the view size.
        m_isScalingViewToFitDocument = true;
        webPage->setUseFixedLayout(true);
        viewScale = (float)viewWidth / (float)documentWidth;
        if (viewScale < minimumViewScale) {
            viewScale = minimumViewScale;
            documentWidth = std::ceil(viewWidth / viewScale);
        }
        IntSize fixedLayoutSize(documentWidth, std::ceil((webPage->size().height() - webPage->corePage()->topContentInset()) / viewScale));
        webPage->setFixedLayoutSize(fixedLayoutSize);

        LOG(Resize, "  using fixed layout at %dx%d. document width %d, scaled to %.4f to fit view width %d", fixedLayoutSize.width(), fixedLayoutSize.height(), documentWidth, viewScale, viewWidth);
    }

    webPage->scaleView(viewScale);
}

void DrawingArea::setShouldScaleViewToFitDocument(bool shouldScaleView)
{
    if (m_shouldScaleViewToFitDocument == shouldScaleView)
        return;

    m_shouldScaleViewToFitDocument = shouldScaleView;
    triggerRenderingUpdate();
}

} // namespace WebKit
