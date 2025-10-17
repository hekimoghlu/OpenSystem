/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#include "PageOverlay.h"

#include "GraphicsContext.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "Logging.h"
#include "Page.h"
#include "PageOverlayController.h"
#include "PlatformMouseEvent.h"
#include "ScrollbarTheme.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

static const Seconds fadeAnimationDuration { 200_ms };
static const double fadeAnimationFrameRate = 30;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageOverlay);

static PageOverlay::PageOverlayID generatePageOverlayID()
{
    static PageOverlay::PageOverlayID pageOverlayID;
    return ++pageOverlayID;
}

Ref<PageOverlay> PageOverlay::create(PageOverlayClient& client, OverlayType overlayType, AlwaysTileOverlayLayer alwaysTileOverlayLayer)
{
    return adoptRef(*new PageOverlay(client, overlayType, alwaysTileOverlayLayer));
}

PageOverlay::PageOverlay(PageOverlayClient& client, OverlayType overlayType, AlwaysTileOverlayLayer alwaysTileOverlayLayer)
    : m_client(client)
    , m_fadeAnimationTimer(*this, &PageOverlay::fadeAnimationTimerFired)
    , m_fadeAnimationDuration(fadeAnimationDuration)
    , m_needsSynchronousScrolling(overlayType == OverlayType::View)
    , m_overlayType(overlayType)
    , m_alwaysTileOverlayLayer(alwaysTileOverlayLayer)
    , m_pageOverlayID(generatePageOverlayID())
{
}

PageOverlay::~PageOverlay() = default;

Page* PageOverlay::page() const
{
    return m_page.get();
}

PageOverlayController* PageOverlay::controller() const
{
    if (!m_page)
        return nullptr;
    return &m_page->pageOverlayController();
}

IntRect PageOverlay::bounds() const
{
    if (!m_overrideFrame.isEmpty())
        return { { }, m_overrideFrame.size() };

    RefPtr frameView = m_page->mainFrame().virtualView();
    if (!frameView)
        return IntRect();

    switch (m_overlayType) {
    case OverlayType::View: {
        int width = frameView->width();
        int height = frameView->height();

        if (!ScrollbarTheme::theme().usesOverlayScrollbars()) {
            if (frameView->verticalScrollbar())
                width -= frameView->verticalScrollbar()->width();
            if (frameView->horizontalScrollbar())
                height -= frameView->horizontalScrollbar()->height();
        }
        return IntRect(0, 0, width, height);
    }
    case OverlayType::Document:
        return IntRect(IntPoint(), frameView->contentsSize());
    }

    ASSERT_NOT_REACHED();
    return IntRect(IntPoint(), frameView->contentsSize());
}

IntRect PageOverlay::frame() const
{
    if (!m_overrideFrame.isEmpty())
        return m_overrideFrame;

    return bounds();
}

void PageOverlay::setFrame(IntRect frame)
{
    if (m_overrideFrame == frame)
        return;

    m_overrideFrame = frame;

    if (auto pageOverlayController = controller())
        pageOverlayController->didChangeOverlayFrame(*this);
}

IntSize PageOverlay::viewToOverlayOffset() const
{
    switch (m_overlayType) {
    case OverlayType::View:
        return IntSize();

    case OverlayType::Document: {
        RefPtr frameView = m_page->mainFrame().virtualView();
        return frameView ? toIntSize(frameView->viewToContents(IntPoint())) : IntSize();
    }
    }
    return IntSize();
}

void PageOverlay::setBackgroundColor(const Color& backgroundColor)
{
    if (m_backgroundColor == backgroundColor)
        return;

    m_backgroundColor = backgroundColor;

    if (auto pageOverlayController = controller())
        pageOverlayController->didChangeOverlayBackgroundColor(*this);
}

void PageOverlay::setPage(Page* page)
{
    m_client.willMoveToPage(*this, page);
    m_page = page;
    m_client.didMoveToPage(*this, page);

    m_fadeAnimationTimer.stop();
}

void PageOverlay::setNeedsDisplay(const IntRect& dirtyRect)
{
    if (auto pageOverlayController = controller()) {
        if (m_fadeAnimationType != FadeAnimationType::NoAnimation)
            pageOverlayController->setPageOverlayOpacity(*this, m_fractionFadedIn);
        pageOverlayController->setPageOverlayNeedsDisplay(*this, dirtyRect);
    }
}

void PageOverlay::setNeedsDisplay()
{
    setNeedsDisplay(bounds());
}

void PageOverlay::drawRect(GraphicsContext& graphicsContext, const IntRect& dirtyRect)
{
    // If the dirty rect is outside the bounds, ignore it.
    IntRect paintRect = intersection(dirtyRect, bounds());
    if (paintRect.isEmpty())
        return;

    GraphicsContextStateSaver stateSaver(graphicsContext);

    if (m_overlayType == PageOverlay::OverlayType::Document) {
        if (auto* frameView = m_page->mainFrame().virtualView()) {
            auto offset = frameView->scrollOrigin();
            graphicsContext.translate(toFloatSize(offset));
            paintRect.moveBy(-offset);
        }
    }

    m_client.drawRect(*this, graphicsContext, paintRect);
}
    
bool PageOverlay::mouseEvent(const PlatformMouseEvent& mouseEvent)
{
    IntPoint mousePositionInOverlayCoordinates(mouseEvent.position());

    if (m_overlayType == PageOverlay::OverlayType::Document)
        mousePositionInOverlayCoordinates = m_page->mainFrame().virtualView()->windowToContents(mousePositionInOverlayCoordinates);
    mousePositionInOverlayCoordinates.moveBy(-frame().location());

    // Ignore events outside the bounds.
    if (m_shouldIgnoreMouseEventsOutsideBounds && !bounds().contains(mousePositionInOverlayCoordinates))
        return false;

    return m_client.mouseEvent(*this, mouseEvent);
}

void PageOverlay::didScrollFrame(LocalFrame& frame)
{
    m_client.didScrollFrame(*this, frame);
}

bool PageOverlay::copyAccessibilityAttributeStringValueForPoint(String attribute, FloatPoint parameter, String& value)
{
    return m_client.copyAccessibilityAttributeStringValueForPoint(*this, attribute, parameter, value);
}

bool PageOverlay::copyAccessibilityAttributeBoolValueForPoint(String attribute, FloatPoint parameter, bool& value)
{
    return m_client.copyAccessibilityAttributeBoolValueForPoint(*this, attribute, parameter, value);
}

Vector<String> PageOverlay::copyAccessibilityAttributeNames(bool parameterizedNames)
{
    return m_client.copyAccessibilityAttributeNames(*this, parameterizedNames);
}

void PageOverlay::startFadeInAnimation()
{
    if (m_fadeAnimationType == FadeInAnimation && m_fadeAnimationTimer.isActive())
        return;

    m_fractionFadedIn = 0;
    m_fadeAnimationType = FadeInAnimation;

    startFadeAnimation();
}

void PageOverlay::startFadeOutAnimation()
{
    if (m_fadeAnimationType == FadeOutAnimation && m_fadeAnimationTimer.isActive())
        return;

    m_fractionFadedIn = 1;
    m_fadeAnimationType = FadeOutAnimation;

    startFadeAnimation();
}

void PageOverlay::stopFadeOutAnimation()
{
    m_fractionFadedIn = 1.0;
    m_fadeAnimationTimer.stop();
}

void PageOverlay::startFadeAnimation()
{
    ASSERT(m_page);
    if (!m_page)
        RELEASE_LOG_FAULT(Animations, "PageOverlay::startFadeAnimation() was called on a PageOverlay without a page");
    m_fadeAnimationStartTime = WallTime::now();
    m_fadeAnimationTimer.startRepeating(1_s / fadeAnimationFrameRate);
}

void PageOverlay::fadeAnimationTimerFired()
{
    auto controller = this->controller();
    ASSERT(controller);

    float animationProgress = (WallTime::now() - m_fadeAnimationStartTime) / m_fadeAnimationDuration;

    if (animationProgress >= 1.0)
        animationProgress = 1.0;

    double sine = sin(piOverTwoFloat * animationProgress);
    float fadeAnimationValue = sine * sine;

    m_fractionFadedIn = (m_fadeAnimationType == FadeInAnimation) ? fadeAnimationValue : 1 - fadeAnimationValue;

    if (controller)
        controller->setPageOverlayOpacity(*this, m_fractionFadedIn);

    if (animationProgress == 1.0) {
        m_fadeAnimationTimer.stop();

        bool wasFadingOut = m_fadeAnimationType == FadeOutAnimation;
        m_fadeAnimationType = NoAnimation;

        // If this was a fade out, uninstall the page overlay.
        if (wasFadingOut && controller)
            controller->uninstallPageOverlay(*this, PageOverlay::FadeMode::DoNotFade);
    }
}

void PageOverlay::clear()
{
    if (auto pageOverlayController = controller())
        pageOverlayController->clearPageOverlay(*this);
}

GraphicsLayer& PageOverlay::layer()
{
    return controller()->layerForOverlay(*this);
}

} // namespace WebKit
